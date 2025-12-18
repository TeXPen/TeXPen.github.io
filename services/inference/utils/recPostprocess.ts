/**
 * Postprocess for PaddleOCR Text Recognition
 * CTC Decode
 */

// Standard English Dict for PaddleOCR (96 keys usually)
// If the model is the multilingual one or specific english one, the dict changes.
// Using a standard printable ascii set as a safe default for "English" models.
// "0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~!"#$%&'()*+,-./ "
const DEFAULT_DICT = "0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~!\"#$%&'()*+,-./ ";

export function recPostprocess(
  data: Float32Array,
  dims: number[], // [1, SequenceLength, NumClasses]
  vocab: string = DEFAULT_DICT
): string {
  // dims: [Batch=1, SeqLen, NumClasses]
  const seqLen = dims[1];
  const numClasses = dims[2];

  const charIndices: number[] = [];

  // ArgMax per time step
  for (let t = 0; t < seqLen; t++) {
    let maxVal = -Infinity;
    let maxIdx = 0;

    const offset = t * numClasses;
    for (let c = 0; c < numClasses; c++) {
      const val = data[offset + c];
      if (val > maxVal) {
        maxVal = val;
        maxIdx = c;
      }
    }
    charIndices.push(maxIdx);
  }

  // CTC Decode: Drop repeats and blanks
  // Blank index is usually last or 0. Need to check Paddle conversion.
  // For standard PaddleOCR:
  // The dict is length N.
  // The model outputs N+1 classes.
  // The blank token is usually at the end (index N).

  // Ref: PaddleOCR defaults blank to last.
  const blankIdx = numClasses - 1;

  let res = "";
  let lastIdx = -1;

  for (const idx of charIndices) {
    if (idx !== lastIdx && idx !== blankIdx) {
      // Append char
      // Be careful with vocab bounds
      if (idx < vocab.length) {
        res += vocab[idx];
      }
    }
    lastIdx = idx;
  }

  return res;
}
