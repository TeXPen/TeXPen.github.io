import { getDB, getPartialDownload, saveChunk, clearPartialDownload, getChunk } from './db';
import { DownloadProgress } from './types';
import { env } from '@huggingface/transformers';

export class DownloadManager {
  private static instance: DownloadManager;
  private activeDownloads: Map<string, Promise<void>> = new Map();
  private queue: Array<() => Promise<void>> = [];
  private runningCount: number = 0;
  private readonly MAX_CONCURRENT = 3;
  private readonly ENABLE_CORRUPTION_CHECK = false;
  // Lower threshold to 10MB for mobile stability (was 50MB)
  private readonly BUFFER_THRESHOLD = 5 * 1024 * 1024; // Further lowered to 5MB

  // Global state for IDB availability across all downloads in this session
  private isIDBDisabled: boolean = false;
  private quotaDialogPromise: Promise<boolean> | null = null;

  private constructor() { }

  public static getInstance(): DownloadManager {
    if (!DownloadManager.instance) {
      DownloadManager.instance = new DownloadManager();
    }
    return DownloadManager.instance;
  }

  private quotaErrorHandler: (() => Promise<boolean>) | null = null;

  public setQuotaErrorHandler(handler: () => Promise<boolean>) {
    this.quotaErrorHandler = handler;
  }

  /**
   * Ensures a file is fully downloaded and cached in the Transformers cache.
   * Resumes from IndexedDB if interrupted.
   */
  public downloadFile(url: string, onProgress?: (progress: DownloadProgress) => void): Promise<void> {
    // Deduplicate concurrent requests for the same URL immediately (synchronous check)
    if (this.activeDownloads.has(url)) {
      return this.activeDownloads.get(url)!;
    }

    // Wrap the download logic in a task
    const task = async () => {
      try {
        // 1. Check if already in browser Cache Storage (transformers.js default location)
        // @ts-expect-error - env.cacheName exists in runtime
        const cacheName = env.cacheName || 'transformers-cache';
        const cache = await caches.open(cacheName);
        const cachedResponse = await cache.match(url);

        if (cachedResponse) {
          // Optimization: Skip corruption check by default
          if (this.ENABLE_CORRUPTION_CHECK) {
            const contentLength = cachedResponse.headers.get('Content-Length');
            const expectedSize = contentLength ? parseInt(contentLength, 10) : 0;
            const actualBlob = await cachedResponse.clone().blob();

            if (expectedSize > 0 && actualBlob.size !== expectedSize) {
              console.warn(`[DownloadManager] Cached file ${url} is corrupted (size mismatch: ${actualBlob.size} vs ${expectedSize}). Re-downloading.`);
              await cache.delete(url);
            } else if (actualBlob.size === 0 && expectedSize > 0) {
              console.warn(`[DownloadManager] Cached file ${url} is empty. Re-downloading.`);
              await cache.delete(url);
            } else {
              return; // Valid cache, skip download
            }
          } else {
            return; // Assume valid cache
          }
        }

        await this._performDownload(url, cache, onProgress);
      } finally {
        this.activeDownloads.delete(url);
        this.runningCount--;
        this._processQueue();
      }
    };

    // Create a controlled promise that resolves when the task actually finishes
    const controlledPromise = new Promise<void>((resolve, reject) => {
      const executeTask = async () => {
        try {
          await task();
          resolve();
        } catch (e) {
          reject(e);
        }
      };

      if (this.runningCount < this.MAX_CONCURRENT) {
        this.runningCount++;
        executeTask();
      } else {
        this.queue.push(executeTask);
      }
    });

    this.activeDownloads.set(url, controlledPromise);
    return controlledPromise;
  }

  // Helper to create a stream from an array of Blobs/Uint8Arrays (Memory Mode)
  private createBlobStream(chunks: (Blob | Uint8Array)[]): ReadableStream {
    let index = 0;
    return new ReadableStream({
      async pull(controller) {
        if (index < chunks.length) {
          const chunk = chunks[index++];
          if (chunk instanceof Blob) {
            const buffer = await new Response(chunk).arrayBuffer();
            controller.enqueue(new Uint8Array(buffer));
          } else {
            controller.enqueue(chunk); // Assuming Uint8Array
          }
        } else {
          controller.close();
        }
      }
    });
  }

  // Helper to create a stream that reads chunks sequentially from IndexedDB (Disk Mode)
  private createIDBStream(url: string, totalChunks: number): ReadableStream {
    let index = 0;

    return new ReadableStream({
      async pull(controller) {
        try {
          if (index < totalChunks) {
            // Read next chunk from IDB
            const chunk = await getChunk(url, index);

            if (!chunk) {
              controller.error(new Error(`Missing chunk ${index} for ${url} in IDB`));
              return;
            }

            const buffer = await new Response(chunk).arrayBuffer();
            controller.enqueue(new Uint8Array(buffer));
            index++;
          } else {
            controller.close();
          }
        } catch (e) {
          controller.error(e);
        }
      }
    });
  }



  private _processQueue() {
    if (this.queue.length > 0 && this.runningCount < this.MAX_CONCURRENT) {
      const nextTask = this.queue.shift();
      if (nextTask) {
        this.runningCount++;
        nextTask();
      }
    }
  }

  private async _performDownload(url: string, cache: Cache, onProgress?: (progress: DownloadProgress) => void): Promise<void> {
    // 1. Proactive IDB availability check - detect early instead of failing on first saveChunk
    // This helps on mobile browsers where IDB is often unavailable (private mode, etc.)
    if (!this.isIDBDisabled) {
      const db = await getDB();
      if (!db) {
        console.warn('[DownloadManager] IndexedDB is unavailable. Falling back to memory-only mode.');
        this.isIDBDisabled = true;
      }
    }

    // 2. Check partial download in IndexedDB (will return null if IDB unavailable)
    const partial = await getPartialDownload(url);
    let startByte = 0;

    // In memory mode, we track chunks here. In Disk mode, we don't.
    // If we are resuming, we don't have the previous chunks in memory anymore (unless we read them, which we shouldn't).
    // So if resuming in Disk Mode, 'downloadedChunks' will be empty, and we will only stream from IDB at the end.

    let chunkIndex = 0;

    if (partial) {
      // partial is now Metadata: { chunkCount, totalBytes, ... }
      console.log(`[DownloadManager] Resuming ${url} from ${partial.chunkCount} chunks...`);
      // We need to calculate startByte. 
      // If we didn't store individual chunk sizes in metadata, we have to trust totalBytes or sum them up?
      // Wait, we don't know the size of existing chunks efficiently unless we read them or store running total.
      // But we know 'totalBytes' in metadata is the *expected* total. That doesn't help with resume position.
      // We need the size of what we have.

      // FIX: accurately calculate startByte by summing sizes of existing chunks.
      // Since we split the store, we have to iterate chunks. This is fast (metadata only usually or small blobs).
      // Or we could have stored 'downloadedBytes' in metadata.
      // For this immediate refactor, let's just sum them up. It's O(chunks), better than O(bytes).

      // Wait, 'getPartialDownload' only returns metadata. 
      // Let's iterate.
      for (let i = 0; i < partial.chunkCount; i++) {
        const chunk = await getChunk(url, i);
        if (chunk) {
          startByte += chunk.size;
        } else {
          // Missing a chunk? Broken state.
          console.warn(`[DownloadManager] Missing chunk ${i} during resume calc. Restarting.`);
          startByte = 0;
          chunkIndex = 0;
          await clearPartialDownload(url);
          break;
        }
      }
      chunkIndex = partial.chunkCount;
      // If we broke loop, startByte is 0.
    }

    // 3. Fetch with Range
    const headers: HeadersInit = {};
    if (startByte > 0) {
      headers['Range'] = `bytes=${startByte}-`;
    }

    const response = await fetch(url, { headers });

    let isComplete = false;

    if (!response.ok) {
      if (response.status === 416) {
        // Handle 416 Range Not Satisfiable
        // This usually means we already have the full file
        const contentRange = response.headers.get('Content-Range');
        if (contentRange) {
          const match = contentRange.match(/\*\/(\d+)/);
          if (match) {
            const serverSize = parseInt(match[1], 10);
            if (startByte === serverSize) {
              console.log(`[DownloadManager] 416 received but local size (${startByte}) matches server size (${serverSize}). Assuming download complete.`);
              isComplete = true;
            } else {
              throw new Error(`Failed to download ${url}: Range Not Satisfiable. Local: ${startByte}, Server: ${serverSize}`);
            }
          }
        } else {
          throw new Error(`Failed to download ${url}: 416 Range Not Satisfiable (no Content-Range header)`);
        }
      } else if (response.status !== 206) {
        throw new Error(`Failed to download ${url}: ${response.status} ${response.statusText}`);
      }
    }

    const contentLengthHeader = response.headers.get('Content-Length');
    // If completed (416), totalSize is startByte. Otherwise calculate.
    const totalSize = isComplete
      ? startByte
      : (contentLengthHeader ? parseInt(contentLengthHeader, 10) + startByte : (partial?.totalBytes || 0));

    const etag = response.headers.get('Etag');

    // If server doesn't support ranges (200 OK instead of 206 Partial Content), we must restart
    if (startByte > 0 && response.status === 200) {
      console.warn('[DownloadManager] Server returned 200 instead of 206. Restarting download from scratch.');
      startByte = 0;
      chunkIndex = 0;
      await clearPartialDownload(url);
    }

    let receivedLength = startByte; // Start from what we have

    // We no longer keep ALL raw chunks in memory to avoid OOM.
    // OPTIMIZATION: If IDB is enabled, we DO NOT populate this array to save RAM.
    const downloadedChunks: (Blob | Uint8Array)[] = this.isIDBDisabled ? [] : [];
    // Wait, if isIDBDisabled, we need to store them!
    // If not, we don't.
    // If we are resuming in Memory Mode (isIDBDisabled=true), we lost previous chunks!
    // Memory mode cannot resume. So downloadedChunks should just start empty.

    // If we were resuming from IDB but now IDB failed and we switched to Memory?
    // We can't easily fetch previous chunks from IDB if it's disabled/unavailable.
    // So falling back to memory usually implies restarting if we can't read partials.
    // But here 'startByte' might be > 0 from when IDB *was* available.
    // If IDB becomes unavailable mid-stream, we have a problem: we have half on disk, half in memory.
    // Simpler approach: If IDB disabled, we only support full download in memory.
    // If startByte > 0 and we are in memory mode, it's risky. But 'isIDBDisabled' checks happen before partial check.

    if (this.isIDBDisabled && startByte > 0) {
      // We can't resume because we can't read old chunks.
      console.warn('[DownloadManager] Cannot resume in memory-only mode. Restarting.');
      startByte = 0;
      receivedLength = 0;
      chunkIndex = 0;
      // headers was already sent with Range... we might need to re-fetch if we can't use this response.
      // But we already got response. 
      // If response is 206, we are missing beginning. We must re-fetch.
      // We'll throw/recurse? Or just failed.
      throw new Error('Cannot resume download without IndexedDB. Please retry.');
    }


    if (!isComplete) {
      const reader = response.body?.getReader();
      if (!reader) throw new Error('Response body is null');


      // Process stream

      // Buffer for saving chunks to IDB less frequently (optimization)
      let pendingChunks: Uint8Array[] = [];
      let pendingSize = 0;

      const flushBuffer = async () => {
        if (pendingChunks.length === 0) return;
        const mergedBlob = new Blob(pendingChunks as BlobPart[]);

        // Save valid chunk to IDB (best effort)
        if (!this.isIDBDisabled) {
          try {
            await saveChunk(url, mergedBlob, totalSize, chunkIndex++, etag);
          } catch (error) {
            console.warn('[DownloadManager] Failed to save chunk to IndexedDB (likely quota exceeded).', error);

            // Check lock
            if (!this.isIDBDisabled) {
              // If another download already triggered the dialog, wait for it
              if (this.quotaDialogPromise) {
                const shouldContinue = await this.quotaDialogPromise;
                if (!shouldContinue) {
                  throw new Error('Download aborted by user due to storage quota limits.');
                }
              } else {
                // We are the first to hit the error
                if (this.quotaErrorHandler) {
                  this.quotaDialogPromise = this.quotaErrorHandler();
                  const shouldContinue = await this.quotaDialogPromise;
                  this.quotaDialogPromise = null; // Release lock

                  if (!shouldContinue) {
                    throw new Error('Download aborted by user due to storage quota limits.');
                  }

                  // User chose to continue in memory -> Disable globally for this session
                  this.isIDBDisabled = true;
                  console.warn('[DownloadManager] Continuing ALL downloads in memory-only mode.');
                } else {
                  console.warn('[DownloadManager] No quota handler set. Defaulting to memory-only mode.');
                  this.isIDBDisabled = true;
                }
              }
            } else {
              // Already disabled, just skip saving
            }

            // RECOVERY: If we switched to memory mode, we must load all PREVIOUS chunks from IDB
            // because subsequent chunks will only be in memory.
            if (this.isIDBDisabled) {
              // chunkIndex was already incremented. The one that failed is (chunkIndex - 1).
              // We need to recover indices 0 to (chunkIndex - 2).
              const chunksToRecover = chunkIndex - 1;
              if (chunksToRecover > 0 && downloadedChunks.length === 0) {
                console.log(`[DownloadManager] Recovering ${chunksToRecover} chunks from IDB for memory fallback...`);
                for (let i = 0; i < chunksToRecover; i++) {
                  const savedChunk = await getChunk(url, i);
                  if (savedChunk) {
                    downloadedChunks.push(savedChunk);
                  } else {
                    console.error(`[DownloadManager] Failed to recover chunk ${i} from IDB during fallback.`);
                    // If we lose a chunk, the file is corrupt. We should probably abort.
                    throw new Error('Critical: Failed to recover saved chunks from storage during memory fallback.');
                  }
                }
              }
            }
          }
        }

        // Add to our final list of blobs - CRITICAL: This must happen even if IDB fails
        if (this.isIDBDisabled) {
          downloadedChunks.push(mergedBlob);
        }

        // Clear RAM buffer
        pendingChunks = [];
        pendingSize = 0;
      };

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          await flushBuffer();
          break;
        }

        if (value) {
          receivedLength += value.length;

          // Don't push to 'chunks' array anymore, just the pending buffer
          pendingChunks.push(value);
          pendingSize += value.byteLength;

          if (pendingSize >= this.BUFFER_THRESHOLD) {
            await flushBuffer();
          }

          if (onProgress) {
            onProgress({
              file: url.split('/').pop() || 'file',
              loaded: receivedLength,
              total: totalSize
            });
          }
        }
      }
    } // End if (!isComplete)

    console.log(`[DownloadManager] Download complete for ${url}. Assembling and caching...`);

    // Validation: Ensure we actually received the full file
    // If complete (416 code path), receivedLength might not be updated, but startByte is totalSize.
    const finalLength = isComplete ? startByte : receivedLength;
    if (totalSize > 0 && finalLength !== totalSize) {
      throw new Error(`Download incomplete for ${url}. Expected ${totalSize} bytes, received ${finalLength}.`);
    }

    // 4. Assemble and store in Cache API
    // OPTIMIZATION: Use streams to avoid creating a massive Blob in memory

    let stream: ReadableStream;
    if (this.isIDBDisabled) {
      // Memory mode: Stream from the array of blobs we kept
      // If we switched mid-stream, this will be partial and fail validation effectively (or create corrupt file).
      // Ideally we should warn or try to read from IDB if check failed.
      stream = this.createBlobStream(downloadedChunks as Blob[]);
    } else {
      // Disk mode: Stream from IDB chunks
      // chunkIndex is the count of chunks we wrote (or skipped).
      stream = this.createIDBStream(url, chunkIndex);
    }

    const fullResponse = new Response(stream, {
      headers: {
        'Content-Length': totalSize.toString(),
        'Content-Type': 'application/octet-stream' // generic
      }
    });

    await cache.put(url, fullResponse);

    // 5. Cleanup partial IDB
    await clearPartialDownload(url);
  }

  /**
   * Checks if a file exists in the cache and if its size matches the Content-Length header.
   * Streams the cached response to avoid loading multi-hundred-MB blobs into memory at once.
   */
  public async checkCacheIntegrity(url: string): Promise<{ ok: boolean, reason?: string, missing?: boolean }> {
    // @ts-expect-error - env.cacheName exists in runtime but is missing from type definitions
    const cacheName = env.cacheName || 'transformers-cache';
    const cache = await caches.open(cacheName);
    const cachedResponse = await cache.match(url);

    if (!cachedResponse) {
      return { ok: false, missing: true, reason: 'File not found in cache' };
    }

    const contentLength = cachedResponse.headers.get('Content-Length');
    const expectedSize = contentLength ? parseInt(contentLength, 10) : 0;
    const actualSize = await this.measureResponseSize(cachedResponse.clone());

    if (actualSize === 0 && expectedSize > 0) {
      return {
        ok: false,
        reason: 'File is empty (0 bytes)'
      };
    } else if (expectedSize > 0 && actualSize !== expectedSize) {
      return {
        ok: false,
        reason: `Size mismatch: expected ${expectedSize}, got ${actualSize}`
      };
    } else if (actualSize === 0) {
      return {
        ok: false,
        reason: 'File is empty (0 bytes)'
      };
    }

    return { ok: true };
  }

  private async measureResponseSize(response: Response): Promise<number> {
    if (!response.body) {
      // Should not happen for cloned responses, but fallback to blob()
      const blob = await response.blob();
      return blob.size;
    }

    const reader = response.body.getReader();
    let total = 0;

    // Stream through the entire body without retaining previous chunks
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        total += value.byteLength;
      }
    }

    reader.releaseLock();
    return total;
  }

  public async deleteFromCache(url: string): Promise<void> {
    // @ts-expect-error - env.cacheName exists in runtime but is missing from type definitions
    const cacheName = env.cacheName || 'transformers-cache';
    const cache = await caches.open(cacheName);
    await cache.delete(url);
    await clearPartialDownload(url);
    console.log(`[DownloadManager] Deleted ${url} from cache and cleared partial download.`);
  }

}

export const downloadManager = DownloadManager.getInstance();
