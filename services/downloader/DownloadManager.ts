
import { ParallelDownloader, ParallelDownloadProgress } from './ParallelDownloader';
import { ChunkStore } from './ChunkStore';
import { env } from '@huggingface/transformers';
import { DownloadProgress } from './types';
import { createSHA256 } from 'hash-wasm';

export class DownloadManager {
  private static instance: DownloadManager;
  private store: ChunkStore;
  private activeDownloads: Map<string, ParallelDownloader> = new Map();
  private queue: Array<{ url: string, onProgress?: (p: DownloadProgress) => void, resolve: () => void, reject: (err: any) => void }> = [];
  private activeCount = 0;
  private MAX_CONCURRENT_FILES = 2; // Mobile friendly limit

  private constructor() {
    this.store = new ChunkStore();
  }

  public static getInstance(): DownloadManager {
    if (!DownloadManager.instance) {
      DownloadManager.instance = new DownloadManager();
    }
    return DownloadManager.instance;
  }

  public setQuotaErrorHandler(handler: () => Promise<boolean>) {
    // TODO: Implement quota handling in V3
    console.warn('Quota handling not yet implemented in V3');
  }

  public async downloadFile(url: string, onProgress?: (progress: DownloadProgress) => void): Promise<void> {
    // 1. Check Cache API first (Legacy compatibility)
    // @ts-expect-error - env.cacheName exists in runtime
    const cacheName = env.cacheName || 'transformers-cache';
    const cache = await caches.open(cacheName);
    const cachedResponse = await cache.match(url);

    if (cachedResponse) {
      const contentLength = cachedResponse.headers.get('Content-Length');
      const expectedSize = contentLength ? parseInt(contentLength, 10) : 0;
      if (expectedSize > 0) {
        onProgress?.({
          loaded: expectedSize,
          total: expectedSize,
          file: url.split('/').pop() || 'unknown'
        });
        return;
      }
    }

    // 2. Queue or Start Download
    await this.scheduleDownload(url, onProgress);

    // 3. Finalize Cache (Move from IDB to Cache API)
    await this.finalizeCache(url, cache);
  }

  private async scheduleDownload(url: string, onProgress?: (p: DownloadProgress) => void): Promise<void> {
    // Deduplication
    if (this.activeDownloads.has(url)) {
      // Existing download, we can't easily hook into the promise of the *current* download 
      // without complex logic. For now, we await it.
      // Ideally we should attach a secondary listener, but ParallelDownloader is simple.
      // Let's just wait for it to finish.
      // NOTE: This doesn't share progress updates with the second caller. 
      // If that's needed, we need an EventEmitter in DownloadManager.
      // Given usage in ModelLoader, usually sequential or handled by UI state.

      // We can poll or wait for the active one to be removed from map.
      while (this.activeDownloads.has(url)) {
        await new Promise(r => setTimeout(r, 100));
      }
      return;
    }

    return new Promise((resolve, reject) => {
      this.queue.push({ url, onProgress, resolve, reject });
      this.processQueue();
    });
  }

  private async processQueue() {
    if (this.activeCount >= this.MAX_CONCURRENT_FILES) return;

    const item = this.queue.shift();
    if (!item) return;

    this.activeCount++;
    const { url, onProgress, resolve, reject } = item;

    const downloader = new ParallelDownloader(url, this.store, {
      onProgress: (p) => {
        onProgress?.({
          loaded: p.loaded,
          total: p.total,
          file: url.split('/').pop() || 'unknown'
        } as any);
      }
    });

    this.activeDownloads.set(url, downloader);

    try {
      await downloader.start();
      this.activeDownloads.delete(url);
      this.activeCount--;
      resolve();
      this.processQueue();
    } catch (err) {
      this.activeDownloads.delete(url);
      this.activeCount--;
      reject(err);
      this.processQueue();
    }
  }

  private async finalizeCache(url: string, cache: Cache) {
    const meta = await this.store.getMetadata(url);
    if (!meta) throw new Error(`Download failed: Metadata missing for ${url}`);

    // Create a stream from chunks
    const stream = await this.store.getStream(url);

    const response = new Response(stream, {
      headers: {
        'Content-Length': meta.totalBytes.toString(),
        'Content-Type': meta.mimeType || 'application/octet-stream'
      }
    });

    await cache.put(url, response);

    // Cleanup IDB after successful cache put?
    // User asked for "Saved checkpoints to allow resuming".
    // If we delete it, we lose the checkpoint if the Cache API decides to evict.
    // However, keeping it doubles storage usage (Cache API + IDB).
    // Standard Transformers.js behavior relies on Cache API.
    // If we want RESUMING, we must keep IDB until we are sure we are done.
    // Once in Cache API, it is "done".
    // If Cache API deletes it, we would have to re-download. 
    // If we keep it in IDB, we can restore.

    // DECISION: Delete from IDB to save space. Checkpointing is for *interrupted* downloads.
    // Once finished, it's effectively "installed".

    await this.store.deleteFile(url);
  }

  public async checkCacheIntegrity(url: string, expectedChecksum?: string): Promise<{ ok: boolean, reason?: string, missing?: boolean }> {
    // @ts-expect-error - env.cacheName exists
    const cacheName = env.cacheName || 'transformers-cache';
    const cache = await caches.open(cacheName);
    const cachedResponse = await cache.match(url);

    if (!cachedResponse) {
      return { ok: false, missing: true, reason: 'File not found in cache' };
    }

    const contentLength = cachedResponse.headers.get('Content-Length');
    if (contentLength) {
      const expectedSize = parseInt(contentLength, 10);
      const blob = await cachedResponse.clone().blob();
      if (blob.size !== expectedSize) {
        return { ok: false, reason: `Size mismatch: expected ${expectedSize}, got ${blob.size}` };
      }

      if (expectedChecksum) {
        try {
          const hasher = await createSHA256();
          hasher.init();

          // Use streaming to avoid loading entire file into memory
          if (blob.stream) {
            const reader = blob.stream().getReader();
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              if (value) {
                hasher.update(value);
              }
            }
          } else {
            // Fallback for environments without blob.stream() (unlikely in modern browsers/Bun)
            // Process in chunks manually if needed, but blob.stream() is standard
            const buffer = await blob.arrayBuffer();
            hasher.update(new Uint8Array(buffer));
          }

          const hashHex = hasher.digest();

          if (hashHex !== expectedChecksum) {
            return { ok: false, reason: `Checksum mismatch: expected ${expectedChecksum}, got ${hashHex}` };
          }
        } catch (error) {
          console.error('Checksum verification failed:', error);
          // Fallback to manual load if stream fails? Or just fail.
          // Let's fail for now to detect issues.
          return { ok: false, reason: `Checksum calculation failed: ${error}` };
        }
      }
    }

    return { ok: true };
  }

  public async deleteFromCache(url: string): Promise<void> {
    // @ts-expect-error - env.cacheName exists
    const cacheName = env.cacheName || 'transformers-cache';
    const cache = await caches.open(cacheName);
    await cache.delete(url);
    await this.store.deleteFile(url);
  }
}

export const downloadManager = DownloadManager.getInstance();
