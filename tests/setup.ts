import { expect, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';
import * as matchers from '@testing-library/jest-dom/matchers';
import { createCanvas, loadImage, Image, ImageData } from '@napi-rs/canvas';

// Extend Vitest's expect method with methods from react-testing-library
expect.extend(matchers as any);
// Aggressively shim ImageData
(global as any).ImageData = ImageData;
(globalThis as any).ImageData = ImageData;
if ((global as any).window) {
  (global as any).window.ImageData = ImageData;
}

// Run cleanup after each test case (e.g. clearing jsdom)
afterEach(() => {
  cleanup();
});

// --- MOCK BROWSER ENVIRONMENT FOR NODE ---

// Minimal Window/Document mock
if (typeof window === 'undefined') {
  (global as any).window = global;
  (global as any).self = global;
  (global as any).window.addEventListener = (type: string, listener: any) => { };
  (global as any).window.removeEventListener = (type: string, listener: any) => { };
  (global as any).sessionStorage = {
    getItem: (key: string) => null,
    setItem: (key: string, value: string) => { },
    removeItem: (key: string) => { },
    clear: () => { }
  };
  (global as any).localStorage = {
    _data: {} as Record<string, string>,
    getItem: function (key: string) { return this._data[key] || null; },
    setItem: function (key: string, value: string) { this._data[key] = value; },
    removeItem: function (key: string) { delete this._data[key]; },
    clear: function () { this._data = {}; }
  };
  (global as any).window.localStorage = (global as any).localStorage;
}

if (typeof document === 'undefined') {
  (global as any).document = {
    createElement: (tagName: string) => {
      if (tagName === 'canvas') return createCanvas(1, 1);
      return {};
    },
    body: {
      appendChild: () => { },
      removeChild: () => { },
    }
  };
}

if (typeof navigator === 'undefined') {
  (global as any).navigator = {
    userAgent: 'node',
    gpu: undefined // Ensure webgpu check fails gracefully or mocks it if needed
  };
}

if (typeof HTMLCanvasElement === 'undefined') {
  // @napi-rs/canvas creates instances that don't strictly inherit from a global HTMLCanvasElement in Node execution 
  // unless we define it. We can just mock checking.
  (global as any).HTMLCanvasElement = class HTMLCanvasElement { };
}

if (typeof OffscreenCanvas === 'undefined') {
  (global as any).OffscreenCanvas = class OffscreenCanvas {
    constructor(width: number, height: number) {
      return createCanvas(width, height) as any;
    }
  } as any;
}

if (typeof Image === 'undefined') {
  (global as any).Image = Image;
}

// MOCK CANVAS
// Properly mock document.createElement to return @napi-rs/canvas canvases for reliable testing
const originalCreateElement = (global as any).document.createElement.bind((global as any).document);
(global as any).document.createElement = (tagName: string, options?: any) => {
  if (tagName.toLowerCase() === 'canvas') {
    const canvas = createCanvas(1, 1) as any;
    // Mock toDataURL
    canvas.toDataURL = (type?: string, quality?: number) => {
      return "data:image/png;base64,mock";
    };
    // Mock toBlob
    canvas.toBlob = (callback: any, type?: any, quality?: any) => {
      callback(new Blob(['mock'], { type: 'image/png' }));
    }
    // Mock getContext to handle willReadFrequently arg which napi-rs might warn about or not support fully matching browser signature
    const originalGetContext = canvas.getContext.bind(canvas);
    canvas.getContext = (contextId: string, options?: any) => {
      return originalGetContext(contextId, options);
    };

    return canvas;
  }
  return originalCreateElement(tagName, options);
};

// Mock createImageBitmap to support real loading via @napi-rs/canvas
(global as any).createImageBitmap = async (blob: any): Promise<ImageBitmap> => {
  let buffer: Buffer;

  // Handle Buffer
  if (Buffer.isBuffer(blob)) {
    buffer = blob;
  }
  // Handle Blob-like objects (duck typing)
  else if (blob && (typeof blob.arrayBuffer === 'function' || blob.size !== undefined || blob.type !== undefined)) {
    try {
      if (typeof blob.arrayBuffer === 'function') {
        const ab = await blob.arrayBuffer();
        buffer = Buffer.from(ab);
      } else {
        // Fallback: Use FileReader
        buffer = await new Promise((resolve, reject) => {
          if (typeof FileReader !== 'undefined') {
            const reader = new FileReader();
            reader.onload = () => {
              if (reader.result instanceof ArrayBuffer) {
                resolve(Buffer.from(reader.result));
              } else {
                reject(new Error("FileReader result wasn't ArrayBuffer"));
              }
            };
            reader.onerror = () => reject(reader.error);
            reader.readAsArrayBuffer(blob);
          } else {
            reject(new Error("Global FileReader not available in Node and arrayBuffer() missing"));
          }
        });
      }
    } catch (e) {
      console.error('[createImageBitmap] Failed to convert blob to buffer:', e);
      throw e;
    }
  } else {
    // Last resort: treat as buffer (e.g. arraybuffer)
    buffer = Buffer.from(blob);
  }

  // Load image using napi-rs
  const img = await loadImage(buffer);

  return {
    width: img.width,
    height: img.height,
    close: () => { },
    _image: img, // Store for drawImage
  } as any;
};

// Monkey-patch the canvas context's drawImage to handle our mock ImageBitmap
// We need to create a dummy canvas to get the prototype.
const dummyCanvas = createCanvas(1, 1);
const ctx = dummyCanvas.getContext('2d');
const CanvasRenderingContext2DPrototype = Object.getPrototypeOf(ctx);

const originalDrawImage = CanvasRenderingContext2DPrototype.drawImage;
CanvasRenderingContext2DPrototype.drawImage = function (image: any, ...args: any[]) {
  // If it's our mock ImageBitmap, use the stored _image
  if (image && image._image) {
    return originalDrawImage.call(this, image._image, ...args);
  }
  return originalDrawImage.call(this, image, ...args);
};
