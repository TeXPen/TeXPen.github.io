
import { ParallelDownloader } from './ParallelDownloader';
import { ChunkStore } from './ChunkStore';
import { env } from '@huggingface/transformers';
import { DownloadProgress } from './types';
import { createSHA256 } from 'hash-wasm';

// @ts-expect-error - env.cacheName exists in runtime
const CACHE_NAME = env.cacheName || 'transformers-cache';

export class DownloadManager {
  private static instance: DownloadManager;
  private store: ChunkStore;
  private activeDownloads: Map<string, ParallelDownloader> = new Map();
  private queue: Array<{ url: string, onProgress?: (p: DownloadProgress) => void, resolve: () => void, reject: (err: unknown) => void }> = [];
  private activeCount = 0;
  private readonly MAX_CONCURRENT_FILES = 2; // Mobile friendly limit
  private quotaErrorHandler: (() => Promise<boolean>) | null = null;
  private quotaErrorAcknowledged = false;

  private constructor() {
    this.store = new ChunkStore();
  }

  private isQuotaError(error: unknown): boolean {
    if (error instanceof DOMException) {
      // QuotaExceededError for Cache API, also check for IndexedDB quota errors
      return error.name === 'QuotaExceededError' ||
        error.code === 22 || // Legacy Safari quota error code
        error.message.includes('quota');
    }
    if (error instanceof Error) {
      return error.message.toLowerCase().includes('quota') ||
        error.message.toLowerCase().includes('storage');
    }
    return false;
  }

  private async handleQuotaError(): Promise<boolean> {
    // Only ask user once per session
    if (this.quotaErrorAcknowledged) {
      return true; // User already acknowledged, continue with current strategy
    }

    if (this.quotaErrorHandler) {
      const shouldContinue = await this.quotaErrorHandler();
      this.quotaErrorAcknowledged = true;
      return shouldContinue;
    }

    // No handler registered, fail gracefully
    console.warn('Quota exceeded and no handler registered');
    return false;
  }

  public static getInstance(): DownloadManager {
    if (!DownloadManager.instance) {
      DownloadManager.instance = new DownloadManager();
    }
    return DownloadManager.instance;
  }

  private getCache(): Promise<Cache> {
    return caches.open(CACHE_NAME);
  }

  private extractFilename(url: string): string {
    return url.split('/').pop() || 'unknown';
  }

  public setQuotaErrorHandler(handler: () => Promise<boolean>) {
    this.quotaErrorHandler = handler;
  }

  public async downloadFile(url: string, onProgress?: (progress: DownloadProgress) => void): Promise<void> {
    const cache = await this.getCache();
    const cachedResponse = await cache.match(url);

    if (cachedResponse) {
      const contentLength = cachedResponse.headers.get('Content-Length');
      const expectedSize = contentLength ? parseInt(contentLength, 10) : 0;
      if (expectedSize > 0) {
        onProgress?.({
          loaded: expectedSize,
          total: expectedSize,
          file: this.extractFilename(url)
        });
        return;
      }
    }

    await this.scheduleDownload(url, onProgress);
    await this.finalizeCache(url, cache);
  }

  private async scheduleDownload(url: string, onProgress?: (p: DownloadProgress) => void): Promise<void> {
    if (this.activeDownloads.has(url)) {
      // Wait for existing download to complete
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

  private async processQueue(): Promise<void> {
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
          file: this.extractFilename(url)
        });
      }
    });

    this.activeDownloads.set(url, downloader);

    try {
      await downloader.start();
      resolve();
    } catch (err) {
      reject(err);
    } finally {
      this.activeDownloads.delete(url);
      this.activeCount--;
      this.processQueue();
    }
  }

  private async finalizeCache(url: string, cache: Cache): Promise<void> {
    const meta = await this.store.getMetadata(url);
    if (!meta) throw new Error(`Download failed: Metadata missing for ${url}`);

    const stream = await this.store.getStream(url);
    const response = new Response(stream, {
      headers: {
        'Content-Length': meta.totalBytes.toString(),
        'Content-Type': meta.mimeType || 'application/octet-stream'
      }
    });

    try {
      await cache.put(url, response);
      await this.store.deleteFile(url);
    } catch (error) {
      if (this.isQuotaError(error)) {
        console.warn(`Quota exceeded when caching ${url}`, error);
        const shouldContinue = await this.handleQuotaError();
        if (!shouldContinue) {
          await this.store.deleteFile(url);
          throw new Error('Download aborted by user: storage quota exceeded');
        }
        // User chose to continue - file is still usable from IndexedDB but won't be in Cache API
        // This means transformers.js won't find it automatically, but the download completed
        console.warn('Continuing without persistent cache storage');
      } else {
        throw error;
      }
    }
  }

  private async computeSha256(blob: Blob): Promise<string> {
    const hasher = await createSHA256();
    hasher.init();

    if (blob.stream) {
      const reader = blob.stream().getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) hasher.update(value);
      }
    } else {
      const buffer = await blob.arrayBuffer();
      hasher.update(new Uint8Array(buffer));
    }

    return hasher.digest();
  }

  public async checkCacheIntegrity(url: string, expectedChecksum?: string): Promise<{ ok: boolean, reason?: string, missing?: boolean }> {
    const cache = await this.getCache();
    const cachedResponse = await cache.match(url);

    if (!cachedResponse) {
      return { ok: false, missing: true, reason: 'File not found in cache' };
    }

    const contentLength = cachedResponse.headers.get('Content-Length');
    if (!contentLength) {
      return { ok: true };
    }

    const expectedSize = parseInt(contentLength, 10);
    const blob = await cachedResponse.clone().blob();

    if (blob.size !== expectedSize) {
      return { ok: false, reason: `Size mismatch: expected ${expectedSize}, got ${blob.size}` };
    }

    if (expectedChecksum) {
      try {
        const hashHex = await this.computeSha256(blob);
        if (hashHex !== expectedChecksum) {
          return { ok: false, reason: `Checksum mismatch: expected ${expectedChecksum}, got ${hashHex}` };
        }
      } catch (error) {
        console.error('Checksum verification failed:', error);
        return { ok: false, reason: `Checksum calculation failed: ${error}` };
      }
    }

    return { ok: true };
  }

  public async cancelDownload(url: string): Promise<void> {
    const downloader = this.activeDownloads.get(url);
    if (downloader) {
      downloader.abort();
      return;
    }

    const index = this.queue.findIndex(item => item.url === url);
    if (index !== -1) {
      const item = this.queue[index];
      this.queue.splice(index, 1);
      item.reject(new Error('Download cancelled'));
    }
  }

  public async deleteFromCache(url: string): Promise<void> {
    await this.cancelDownload(url);
    const cache = await this.getCache();
    await cache.delete(url);
    await this.store.deleteFile(url);
  }
  public async clearCache(): Promise<void> {
    // 1. Cancel all active downloads
    for (const url of this.activeDownloads.keys()) {
      await this.cancelDownload(url);
    }
    this.queue = [];

    // 2. Clear Cache API
    const cache = await this.getCache();
    const keys = await cache.keys();
    for (const request of keys) {
      await cache.delete(request);
    }

    // 3. Clear IndexedDB
    await this.store.clearAll();
  }
}

export const downloadManager = DownloadManager.getInstance();
