
import { openDB, IDBPDatabase, DBSchema } from 'idb';

interface DownloadDB extends DBSchema {
  metadata: {
    key: string; // URL
    value: {
      url: string;
      totalBytes: number;
      mimeType: string;
      etag: string | null;
      lastModified: number;
      chunkSize: number;
      totalChunks: number; // Derived from totalBytes / chunkSize
      chunks: boolean[]; // Bitmask equivalent to track downloaded chunks
    };
  };
  chunks: {
    key: [string, number]; // [URL, ChunkIndex]
    value: Blob;
  };
}

export class ChunkStore {
  private static DB_NAME = 'texpen-downloads-v3';
  private static DB_VERSION = 2;
  private dbPromise: Promise<IDBPDatabase<DownloadDB>>;

  constructor() {
    this.dbPromise = openDB<DownloadDB>(ChunkStore.DB_NAME, ChunkStore.DB_VERSION, {
      upgrade(db, oldVersion) {
        if (!db.objectStoreNames.contains('metadata')) {
          db.createObjectStore('metadata', { keyPath: 'url' });
        }

        if (db.objectStoreNames.contains('chunks')) {
          // If upgrading from v1 where chunks had keyPath, we must delete it
          // because we cannot change keyPath in place.
          // Note: This wipes existing partial downloads, which is acceptable for a bug fix.
          if (oldVersion < 2) {
            db.deleteObjectStore('chunks');
          }
        }

        if (!db.objectStoreNames.contains('chunks')) {
          // Create without keyPath to allow out-of-line keys (put(value, key))
          db.createObjectStore('chunks');
        }
      },
    });
  }

  public async getMetadata(url: string) {
    const db = await this.dbPromise;
    return await db.get('metadata', url);
  }

  public async initFile(url: string, totalBytes: number, mimeType: string, etag: string | null, chunkSize: number) {
    const db = await this.dbPromise;
    const tx = db.transaction('metadata', 'readwrite');
    const store = tx.objectStore('metadata');

    const existing = await store.get(url);
    const totalChunks = Math.ceil(totalBytes / chunkSize);

    if (existing) {
      // Check for modification
      if (existing.etag !== etag || existing.totalBytes !== totalBytes) {
        // File changed on server, reset
        console.warn(`[ChunkStore] File changed for ${url}, resetting.`);
        // Note: Caller is responsible for clearing chunks if this happens, 
        // or we can do it here but that requires multiple object stores transaction.
        // For simplicity, we just overwrite metadata here. Caller should call clear() before init if they suspect mismatch.
      } else {
        // Compatible, return existing
        return existing;
      }
    }

    const meta = {
      url,
      totalBytes,
      mimeType,
      etag,
      lastModified: Date.now(),
      chunkSize,
      totalChunks,
      chunks: new Array(totalChunks).fill(false)
    };

    await store.put(meta);
    await tx.done;
    return meta;
  }

  public async saveChunk(url: string, index: number, data: Blob) {
    const db = await this.dbPromise;
    const tx = db.transaction(['metadata', 'chunks'], 'readwrite');

    // Save blob
    await tx.objectStore('chunks').put(data, [url, index]);

    // Update metadata
    const metaStore = tx.objectStore('metadata');
    const meta = await metaStore.get(url);
    if (meta) {
      if (!meta.chunks[index]) {
        meta.chunks[index] = true;
        meta.lastModified = Date.now();
        await metaStore.put(meta);
      }
    }

    await tx.done;
  }

  public async getChunk(url: string, index: number): Promise<Blob | undefined> {
    const db = await this.dbPromise;
    return await db.get('chunks', [url, index]);
  }

  public async deleteFile(url: string) {
    const db = await this.dbPromise;
    const tx = db.transaction(['metadata', 'chunks'], 'readwrite');

    await tx.objectStore('metadata').delete(url);

    // Delete chunks - efficient range deletion
    const chunkStore = tx.objectStore('chunks');
    const range = IDBKeyRange.bound([url, 0], [url, Infinity]);
    let cursor = await chunkStore.openCursor(range);
    while (cursor) {
      await cursor.delete();
      cursor = await cursor.continue();
    }

    await tx.done;
  }

  public async clearAll() {
    const db = await this.dbPromise;
    const tx = db.transaction(['metadata', 'chunks'], 'readwrite');
    await tx.objectStore('metadata').clear();
    await tx.objectStore('chunks').clear();
    await tx.done;
  }

  public async getStream(url: string): Promise<ReadableStream<Uint8Array>> {
    const db = await this.dbPromise;
    const meta = await db.get('metadata', url);
    if (!meta) throw new Error(`Missing metadata for ${url}`);

    let index = 0;
    const totalChunks = meta.totalChunks;

    return new ReadableStream({
      async pull(controller) {
        if (index >= totalChunks) {
          controller.close();
          return;
        }

        try {
          const chunkBlob = await db.get('chunks', [url, index]);
          if (!chunkBlob) {
            controller.error(new Error(`Missing chunk ${index} for ${url}`));
            return;
          }
          const buffer = await chunkBlob.arrayBuffer();
          controller.enqueue(new Uint8Array(buffer));
          index++;
        } catch (e) {
          controller.error(e);
        }
      }
    });
  }
}
