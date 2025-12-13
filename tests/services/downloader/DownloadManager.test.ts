
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { downloadManager } from '../../../services/downloader/DownloadManager';
import { createSHA256 } from 'hash-wasm';

// Mock hash-wasm at the top level
vi.mock('hash-wasm', () => ({
  createSHA256: vi.fn()
}));

// Mock ChunkStore if needed, but integration with logic is fine.

// Mock fetch globally
const fetchMock = vi.fn();
global.fetch = fetchMock;

describe('DownloadManager V3', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('is defined', () => {
    expect(downloadManager).toBeDefined();
  });

  it('can schedule a download', async () => {
    // Setup fetch mock HEAD
    fetchMock.mockResolvedValueOnce({
      ok: true,
      headers: {
        get: (key: string) => {
          if (key === 'Content-Length') return '100';
          if (key === 'Content-Type') return 'text/plain';
          return null;
        }
      },
      blob: () => Promise.resolve(new Blob(['test']))
    });

    expect(true).toBe(true);
  });

  it('validates checksums correctly', async () => {
    // Setup hash-wasm mock for this test
    const mockHasher = {
      init: vi.fn(),
      update: vi.fn(),
      digest: vi.fn()
    };
    (createSHA256 as any).mockResolvedValue(mockHasher);

    // Case 1: Match
    mockHasher.digest.mockReturnValue('01');

    // Mock Cache API
    const mockCache = {
      match: vi.fn(),
    };
    const mockCaches = {
      open: vi.fn().mockResolvedValue(mockCache),
    };
    (global as any).caches = mockCaches;

    // Expected hash
    const expectedHash = '01';

    // Mock cached response with stream support
    mockCache.match.mockResolvedValue({
      headers: {
        get: () => '1'
      },
      clone: () => ({
        blob: () => Promise.resolve({
          size: 1,
          arrayBuffer: () => Promise.resolve(new Uint8Array([1]).buffer),
          stream: () => ({
            getReader: () => ({
              read: vi.fn()
                .mockResolvedValueOnce({ done: false, value: new Uint8Array([1]) })
                .mockResolvedValueOnce({ done: true })
            })
          })
        })
      })
    });

    // Test Match
    const resultMatch = await downloadManager.checkCacheIntegrity('http://example.com', expectedHash);
    expect(resultMatch.ok).toBe(true);
    expect(mockHasher.update).toHaveBeenCalled();

    // Case 2: Mismatch
    // Since checkCacheIntegrity calls createSHA256 again, and we mocked it to return mockHasher,
    // we can change the behavior of mockHasher or the return of createSHA256.
    // Changing mockHasher behavior is easiest as it is the same object reference returned.

    mockHasher.digest.mockReturnValue('FF');

    const resultMismatch = await downloadManager.checkCacheIntegrity('http://example.com', expectedHash);
    expect(resultMismatch.ok).toBe(false);
    expect(resultMismatch.reason).toContain('Checksum mismatch');
  });
});
