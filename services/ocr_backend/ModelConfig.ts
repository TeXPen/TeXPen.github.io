export const PADDLE_MODEL_CONFIG = {
  LAYOUT: {
    MODEL_PATH: '/models/paddle/structure_v3/layout.onnx',
    LABELS: [
      "Paragraph Title",
      "Image",
      "Text",
      "Number",
      "Abstract",
      "Content",
      "Figure Caption",
      "Formula",
      "Table",
      "Table Caption",
      "References",
      "Document Title",
      "Footnote",
      "Header",
      "Algorithm",
      "Footer",
      "Seal"
    ]
  },
  DET: {
    MODEL_PATH: '/models/paddle/det.onnx',
    LIMIT_SIDE_LEN: 960
  },
  REC: {
    MODEL_PATH: '/models/paddle/rec.onnx',
    KEY_FILE: '/models/paddle/keys_v5.txt', // V5 Dictionary
    IMG_H: 48,
    IMG_W: 320
  }
};

// Standard PP-OCRv4 English Dictionary (subset)
// Ideally this should be loaded from a file
export const DEFAULT_REC_KEYS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ";
