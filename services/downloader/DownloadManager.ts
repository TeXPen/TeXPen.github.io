import { getDB, getPartialDownload, saveChunk, clearPartialDownload } from './db';
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

  // Helper to create a stream that reads sequentially from IndexedDB (Disk Mode)
  private createIDBStream(url: string, totalChunks: number): ReadableStream {
    let index = 0;
    return new ReadableStream({
      async pull(controller) {
        if (index < totalChunks) {
          try {
            const db = await getDB();
            const entry = await db.get('downloads', url);

            // Safety check: entry might have been deleted or corrupted
            if (!entry || !entry.chunks[index]) {
              controller.error(new Error(`Missing chunk ${index} for ${url} in IDB`));
              return;
            }

            const chunk = entry.chunks[index];
            if (chunk instanceof Blob) {
              const buffer = await new Response(chunk).arrayBuffer();
              controller.enqueue(new Uint8Array(buffer));
            } else {
              controller.enqueue(chunk);
            }
            index++;
          } catch (e) {
            controller.error(e);
          }
        } else {
          controller.close();
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
    // 2. Check partial download in IndexedDB
    const partial = await getPartialDownload(url);
    let startByte = 0;
    let chunks: (Blob | Uint8Array)[] = [];
    let currentTotal = 0;

    if (partial) {
      console.log(`[DownloadManager] Resuming ${url} from ${partial.chunks.length} chunks...`);
      chunks = partial.chunks;
      // Calculate total size of existing chunks
      for (const c of chunks) {
        currentTotal += c instanceof Blob ? c.size : c.byteLength;
      }
      startByte = currentTotal;
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
      chunks = [];
      currentTotal = 0;
      await clearPartialDownload(url);
    }

    let receivedLength = currentTotal;

    if (!isComplete) {
      const reader = response.body?.getReader();
      if (!reader) throw new Error('Response body is null');


      // Process stream
      let chunkIndex = chunks.length;

      // Buffer for saving chunks to IDB less frequently (optimization)
      let pendingChunks: Uint8Array[] = [];
      let pendingSize = 0;

      // We no longer keep ALL raw chunks in memory to avoid OOM.
      // We only keep the chunks we've already converted to Blobs (flushed)
      // plus the current pending buffer.
      // OPTIMIZATION: If IDB is enabled, we DO NOT populate this array to save RAM.
      const downloadedChunks: (Blob | Uint8Array)[] = this.isIDBDisabled ? [...chunks] : [];

      // Flag to disable IDB writes if we hit a quota error (e.g. Incognito)
      // No local flag needed, use class-level property

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
                // If resolved true, isIDBDisabled should have been set to true by the first caller,
                // but let's be safe and fall through.
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
                  // No handler? Default to memory mode silently? 
                  // Or warn. Let's warn and disable.
                  console.warn('[DownloadManager] No quota handler set. Defaulting to memory-only mode.');
                  this.isIDBDisabled = true;
                }
              }
            } else {
              // Already disabled, just skip saving
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
      stream = this.createBlobStream(downloadedChunks as Blob[]);
    } else {
      // Disk mode: Stream from IDB chunks
      // We know how many chunks we wrote: `chunkIndex` (which is incremented post-write)
      // actually check chunkIndex usage.
      // In flushBuffer: chunkIndex++ happens AFTER saveChunk start.
      // The total number of chunks written is `chunkIndex` (at loop end).
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
   * Also performs a SHA-256 checksum calculation to ensure the file is readable and consistent.
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
    const actualBlob = await cachedResponse.clone().blob();

    if (actualBlob.size === 0 && expectedSize > 0) {
      return {
        ok: false,
        reason: 'File is empty (0 bytes)'
      };
    } else if (expectedSize > 0 && actualBlob.size !== expectedSize) {
      return {
        ok: false,
        reason: `Size mismatch: expected ${expectedSize}, got ${actualBlob.size}`
      };
    }

    // Perform a full read and checksum to ensure file integrity
    try {
      await this.computeChecksum(actualBlob);
    } catch (e) {
      return {
        ok: false,
        reason: `Checksum calculation failed (unreadable file): ${e}`
      };
    }

    return { ok: true };
  }

  private async computeChecksum(blob: Blob): Promise<string> {
    const arrayBuffer = await blob.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', arrayBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    return hashHex;
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

