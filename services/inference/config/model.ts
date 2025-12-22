export const MODEL_CONFIG = {
  ID: 'Ji-Ha/TexTeller3-ONNX-dynamic',
  DEFAULT_PROVIDER: 'wasm',

  // Models
  PADDLE_VL_ID: 'PaddlePaddle/PaddleOCR-VL',
  PADDLE_VL_ONNX_REPO: 'USER_NAME/REPO_NAME', // Updated by user after upload
  LATEX_DET_ID: 'breezedeus/pix2text-mfd',
  // TEXT_DETECTOR_ID: 'monkt/paddleocr-onnx', // Deprecated
  // TEXT_RECOGNIZER_ID: 'monkt/paddleocr-onnx', // Deprecated

  // Filenames
  PADDLE_VL_ENCODER: 'encoder_model.onnx',
  PADDLE_VL_DECODER: 'decoder_model_merged.onnx',
  LATEX_DET_MODEL: 'mfd-v20240618.onnx',

  IMAGE_SIZE: 448,
  MEAN: [0.9545467],
  STD: [0.15394445],

  // Input/Output Names
  ENCODER_INPUT_NAME: 'pixel_values',
  DECODER_INPUT_NAME: 'decoder_input_ids',
  DECODER_OUTPUT_NAME: 'logits',

  // Special Tokens
  TOKENS: {
    EOS: '</s>',
    BOS: '<s>',
    PAD: '<pad>',
  },

  // Environment / Backend
  PROVIDERS: {
    WEBGPU: 'webgpu',
    WASM: 'wasm',
  },
  CHECKSUMS: {
    'encoder_model.onnx': '5e19cbcea4a6e28c3c4a6e52aca380e2f6e59a463a8c0df8330927b97fdc5499',
    'decoder_with_past_model.onnx': '30bfb67fcfe25055c85c0421ca7b1da608730048bc72ff191c7394e66f780f94',
    // Add checksums for new models if available/needed
  },
} as const;
