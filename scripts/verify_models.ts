import * as ort from 'onnxruntime-node';
import path from 'path';

const MODELS_DIR = path.join(process.cwd(), 'public', 'models', 'paddle');

async function inspectModel(name: string, inputShape: number[]) {
  const modelPath = path.join(MODELS_DIR, name);
  console.log(`\n--- Inspecting ${name} ---`);
  try {
    const session = await ort.InferenceSession.create(modelPath);
    console.log(`Loaded.`);

    // Inputs
    const inputName = session.inputNames[0];
    console.log(`Input Name: ${inputName}`);

    // Create dummy tensor
    // Shape: [1, 3, H, W]
    const size = inputShape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size).fill(0.5);
    const tensor = new ort.Tensor('float32', data, inputShape);

    // Run
    const feeds = { [inputName]: tensor };
    const results = await session.run(feeds);

    const outputName = session.outputNames[0];
    console.log(`Output Name: ${outputName}`);
    const output = results[outputName];
    console.log(`Output Shape: ${output.dims}`);

  } catch (e) {
    console.error(`Failed to test ${name}:`, e);
  }
}

async function main() {
  // Det: [1, 3, 640, 640] - standard check
  await inspectModel('det.onnx', [1, 3, 640, 640]);

  // Rec: [1, 3, 48, 320]
  await inspectModel('rec.onnx', [1, 3, 48, 320]);
}

main();
