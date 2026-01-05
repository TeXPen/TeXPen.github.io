import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PaddleOCREngine } from '../../../services/ocr_backend/PaddleOCREngine';
import * as ort from 'onnxruntime-web';

// Mock ONNX Runtime
vi.mock('onnxruntime-web', () => {
  return {
    InferenceSession: {
      create: vi.fn(),
    },
    Tensor: class {
      constructor(public type: string, public data: any, public dims: number[]) { }
    }
  };
});

describe('PaddleOCREngine', () => {
  let engine: PaddleOCREngine;
  let mockLayoutSession: any;
  let mockDetSession: any;
  let mockRecSession: any;

  beforeEach(() => {
    vi.clearAllMocks();
    engine = new PaddleOCREngine();

    // Create mock sessions with run methods
    mockLayoutSession = {
      inputNames: ['images'],
      outputNames: ['output'],
      run: vi.fn()
    };
    mockDetSession = {
      inputNames: ['images'],
      outputNames: ['output'],
      run: vi.fn()
    };
    mockRecSession = {
      inputNames: ['images'],
      outputNames: ['output'],
      run: vi.fn()
    };

    // Mock InferenceSession.create to return our mock sessions in order
    // Order: Layout, Det, Rec
    (ort.InferenceSession.create as any)
      .mockResolvedValueOnce(mockLayoutSession)
      .mockResolvedValueOnce(mockDetSession)
      .mockResolvedValueOnce(mockRecSession);

    // Mock fetch for keys file
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      text: () => Promise.resolve("a\nb\nc\nd\ne\nf\ng\n") // minimal keys
    });
  });

  it('should initialize correctly', async () => {
    await engine.init();
    expect(ort.InferenceSession.create).toHaveBeenCalledTimes(3);
  });

  it('should process an image and return structured results', async () => {
    // Setup Mock Responses

    // 1. Layout Response
    // Shape: [N, 6] -> [class, score, x1, y1, x2, y2]
    // Let's return 2 layout items:
    // Item 1: Title (Class 0 in ModelConfig is "Paragraph Title"?) 
    // Wait, ModelConfig LABELS: 0=Paragraph Title, 2=Text
    // Let's say we have a Title at 0,0,100,50
    // and Text at 0,60,100,100
    // Standard output is flattened Float32Array
    const layoutOutputData = new Float32Array([
      0, 0.9, 0, 0, 100, 50,    // Title
      2, 0.9, 0, 60, 100, 150   // Text
    ]);
    mockLayoutSession.run.mockResolvedValue({
      'output': {
        data: layoutOutputData,
        dims: [1, 12] // 2 boxes * 6
      }
    });

    // 2. Det Response (called for each Text region)
    // We have 2 text regions (Title calls Det, Text calls Det)
    // Let's blindly return 1 box for each region for simplicity.
    // Det output: [1, 1, H, W] sigmoid map.
    // boxesFromBitmap logic is complex to mock via bitmap.
    // Ideally we should mock `inferDet` private method or just ensure logic flows.
    // If I mock `detSession.run` I have to provide a valid bitmap that produces boxes.
    // That's hard. 
    // ALTERNATIVE: Spy on private methods `inferLayout`, `detectAndRecognize`?
    // But testing public API is better.

    // Let's try to mock the *result* of inferDet/inferRec by partial mocking if possible, 
    // OR just construct a simple bitmap that `boxesFromBitmap` will approve.
    // `boxesFromBitmap` looks for pixels > threshold in the map.

    // Let's just make a huge map of 1s in the center.
    // H, W from Det resizing is 960 max side len.
    // Let's say input is small 100x100. Resized is small.
    // Let's mock a simpler approach: spy on the private methods using `vi.spyOn(engine as any, 'method')`.

    const inferLayoutSpy = vi.spyOn(engine as any, 'inferLayout');
    inferLayoutSpy.mockResolvedValue([
      { type: 'Title', box: [0, 0, 100, 50], confidence: 0.9 },
      { type: 'Text', box: [0, 60, 100, 150], confidence: 0.8 }
    ]);

    const detectAndRecognizeSpy = vi.spyOn(engine as any, 'detectAndRecognize');
    detectAndRecognizeSpy.mockResolvedValue([
      { text: "Hello", confidence: 0.99, box: { points: [[0, 0], [10, 0], [10, 10], [0, 10]] } }
    ]);

    // Create a dummy valid PNG blob (1x1 pixel)
    const base64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKwMTQAAAABJRU5ErkJggg==';
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: 'image/png' });

    const result = await engine.process(blob);

    expect(inferLayoutSpy).toHaveBeenCalled();
    expect(detectAndRecognizeSpy).toHaveBeenCalledTimes(2); // Once for Title, once for Text

    expect(result.layout.length).toBe(2);
    expect(result.layout[0].type).toBe('Title');
    expect(result.layout[0].text).toBe('Hello');
    expect(result.layout[1].type).toBe('Text');
    expect(result.rawText).toContain('Hello');
  });
});
