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
  private readonly BUFFER_THRESHOLD = 10 * 1024 * 1024;

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
        // @ts-ignore - env.cacheName exists in runtime
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

    if (!response.ok && response.status !== 206) {
      throw new Error(`Failed to download ${url}: ${response.status} ${response.statusText}`);
    }

    const contentLengthHeader = response.headers.get('Content-Length');
    const totalSize = contentLengthHeader ? parseInt(contentLengthHeader, 10) + startByte : (partial?.totalBytes || 0);
    const etag = response.headers.get('Etag');

    // If server doesn't support ranges (200 OK instead of 206 Partial Content), we must restart
    if (startByte > 0 && response.status === 200) {
      console.warn('[DownloadManager] Server returned 200 instead of 206. Restarting download from scratch.');
      startByte = 0;
      chunks = [];
      currentTotal = 0;
      await clearPartialDownload(url);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('Response body is null');

    let receivedLength = currentTotal;

    // Process stream
    let chunkIndex = chunks.length;

    // Buffer for saving chunks to IDB less frequently (optimization)
    let pendingChunks: Uint8Array[] = [];
    let pendingSize = 0;

    // We no longer keep ALL raw chunks in memory to avoid OOM.
    // We only keep the chunks we've already converted to Blobs (flushed)
    // plus the current pending buffer.
    const downloadedChunks: (Blob | Uint8Array)[] = [...chunks];

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
      downloadedChunks.push(mergedBlob);

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

    console.log(`[DownloadManager] Download complete for ${url}. Assembling and caching...`);

    // Validation: Ensure we actually received the full file
    if (totalSize > 0 && receivedLength !== totalSize) {
      throw new Error(`Download incomplete for ${url}. Expected ${totalSize} bytes, received ${receivedLength}.`);
    }

    // 4. Assemble and store in Cache API
    // downloadedChunks now contains only Blobs (or initial Uint8Arrays from partial resume)
    const fullBlob = new Blob(downloadedChunks as BlobPart[], { type: 'application/octet-stream' });

    // Release memory held by chunks array immediately
    // chunks reference is local and will be GC'd, downloadedChunks is used for fullBlob then dropped

    const fullResponse = new Response(fullBlob, {
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
   * Returns { ok: true } if valid or missing (cannot verify missing),
   * or { ok: false, reason: string } if corrupted.
   * Note: Missing files are considered "not corrupted" by this check itself,
   * but the caller should verify existence if needed.
   * Actually, let's return a specific status.
   */
  public async checkCacheIntegrity(url: string): Promise<{ ok: boolean, reason?: string, missing?: boolean }> {
    // @ts-ignore
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

    return { ok: true };
  }

  public async deleteFromCache(url: string): Promise<void> {
    // @ts-ignore
    const cacheName = env.cacheName || 'transformers-cache';
    const cache = await caches.open(cacheName);
    await cache.delete(url);
    console.log(`[DownloadManager] Deleted ${url} from cache.`);
  }

}

export const downloadManager = DownloadManager.getInstance();
