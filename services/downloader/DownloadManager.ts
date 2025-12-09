import { getDB, getPartialDownload, saveChunk, clearPartialDownload } from './db';
import { DownloadProgress } from './types';

export class DownloadManager {
  private static instance: DownloadManager;
  private activeDownloads: Map<string, Promise<void>> = new Map();

  private constructor() { }

  public static getInstance(): DownloadManager {
    if (!DownloadManager.instance) {
      DownloadManager.instance = new DownloadManager();
    }
    return DownloadManager.instance;
  }

  /**
   * Ensures a file is fully downloaded and cached in the Transformers cache.
   * Resumes from IndexedDB if interrupted.
   */
  public async downloadFile(url: string, onProgress?: (progress: DownloadProgress) => void): Promise<void> {
    // 1. Check if already in browser Cache Storage (transformers.js default location)
    const cache = await caches.open('transformers-cache');
    const cachedResponse = await cache.match(url);

    if (cachedResponse) {
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
        return; // Valid cache
      }
    }

    // Deduplicate concurrent requests for the same URL
    if (this.activeDownloads.has(url)) {
      return this.activeDownloads.get(url)!;
    }

    const downloadPromise = this._performDownload(url, cache, onProgress).finally(() => {
      this.activeDownloads.delete(url);
    });

    this.activeDownloads.set(url, downloadPromise);
    return downloadPromise;
  }

  private async _performDownload(url: string, cache: Cache, onProgress?: (progress: DownloadProgress) => void): Promise<void> {
    // 2. Check partial download in IndexedDB
    const partial = await getPartialDownload(url);
    let startByte = 0;
    let chunks: Blob[] = [];
    let currentTotal = 0;

    if (partial) {
      console.log(`[DownloadManager] Resuming ${url} from ${partial.chunks.length} chunks...`);
      chunks = partial.chunks;
      // Calculate total size of existing chunks
      for (const c of chunks) {
        currentTotal += c.size;
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
    let pendingChunks: Blob[] = [];
    let pendingSize = 0;
    const BUFFER_THRESHOLD = 50 * 1024 * 1024; // 50MB

    const flushBuffer = async () => {
      if (pendingChunks.length === 0) return;
      const mergedBlob = new Blob(pendingChunks);
      await saveChunk(url, mergedBlob, totalSize, chunkIndex++, etag);
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
        const blob = new Blob([value]);

        chunks.push(blob);
        pendingChunks.push(blob);
        pendingSize += value.byteLength;

        if (pendingSize >= BUFFER_THRESHOLD) {
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

    // 4. Assemble and store in Cache API
    const fullBlob = new Blob(chunks, { type: 'application/octet-stream' });
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
}

export const downloadManager = DownloadManager.getInstance();
