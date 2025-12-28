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
  PADDLE_VL_ENCODER: 'vision_transformer.onnx', // Keeping for compat, but should use VLM_COMPONENTS
  PADDLE_VL_DECODER: 'llm.onnx',

  VLM_COMPONENTS: {
    VISION_PATCH_EMBED: 'vision_patch_embed.onnx',
    VISION_TRANSFORMER: 'vision_transformer.onnx',
    VISION_PROJECTOR: 'vision_projector.onnx',
    TEXT_EMBED: 'text_embed.onnx',
    LLM: 'llm.onnx',
    POS_EMBED: 'pos_embed.npy'
  },

  // Quantization Settings
  // Quantization Settings
  QUANTIZED: true, // Enable to use quantized variants for large models
  QUANTIZED_SUFFIX: '_q4.onnx', // Suffix for quantized files. could be '_awq.onnx' if using AutoAWQ export

  LATEX_DET_MODEL: 'mfd-v20240618.onnx',

  IMAGE_SIZE: 378,
  MEAN: [0.5, 0.5, 0.5],
  STD: [0.5, 0.5, 0.5],

  // Input/Output Names
  ENCODER_INPUT_NAME: 'pixel_values',
  ENCODER_OUTPUT_NAME: 'image_embeds',
  DECODER_INPUT_NAME: 'input_ids',
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
