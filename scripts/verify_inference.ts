
import { JSDOM } from 'jsdom';
import { inferenceService } from '../services/inference/InferenceService';
import fs from 'fs';
import path from 'path';
import { createCanvas, loadImage } from '@napi-rs/canvas';

// Mock Browser Environment
const dom = new JSDOM('<!DOCTYPE html>');
global.document = dom.window.document;
global.window = dom.window as any;
global.HTMLCanvasElement = dom.window.HTMLCanvasElement;
global.ImageData = dom.window.ImageData;
// Don't use JSDOM's Blob - use Bun's native Blob which supports arrayBuffer()
// global.Blob = dom.window.Blob;

// Mock Canvas API using '@napi-rs/canvas' for Node.js
// We need to patch document.createElement to return node-canvas for 'canvas'
const originalCreateElement = global.document.createElement.bind(global.document);
global.document.createElement = (tagName: string, options?: any) => {
  if (tagName.toLowerCase() === 'canvas') {
    const canvas = createCanvas(1, 1);
    // Add missing properties/methods if needed
    return canvas as any;
  }
  return originalCreateElement(tagName, options);
};

// Mock createImageBitmap
global.createImageBitmap = async (blob: any): Promise<ImageBitmap> => {
  const buffer = Buffer.from(await (blob as any).arrayBuffer());
  const img = await loadImage(buffer);
  return img as any;
};

async function runTest() {
  const imagePath = path.resolve(__dirname, '../public/test.png');
  console.log(`Testing with image: ${imagePath}`);

  if (!fs.existsSync(imagePath)) {
    console.error('Test image not found!');
    process.exit(1);
  }

  const buffer = fs.readFileSync(imagePath);
  // Create a Blob-like object
  const blob = new Blob([buffer]);

  console.log('Initializing model...');
  // Use 'cpu' or 'wasm' for Node environment
  await inferenceService.init((status) => console.log(status), { device: 'wasm', dtype: 'fp32' });

  console.log('Running inference...');
  try {
    const result = await inferenceService.infer(blob);
    console.log('\n--- Result ---');
    console.log(result);
    console.log('--------------\n');

    const expected = String.raw`\begin{split}
A&=\frac{\pi r^{2}}{2}\\
&=\frac{1}{2}\pi r^{2}
\end{split}
\qquad\qquad\qquad\qquad \qquad\qquad\qquad(1)`;

    // Simple normalization for comparison (ignore whitespace differences)
    const normalize = (s: string) => s.replace(/\s+/g, ' ').trim();

    if (normalize(result.latex) === normalize(expected)) {
      console.log('✅ Test PASSED!');
    } else {
      console.log('❌ Test FAILED!');
      console.log('Expected:');
      console.log(expected);
    }

  } catch (error) {
    console.error('Inference failed:', error);
  } finally {
    console.log('Cleaning up...');
    await inferenceService.dispose();
    console.log('Done.');
  }
}

runTest();
