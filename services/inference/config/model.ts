export const MODEL_CONFIG = {
  ID: 'Ji-Ha/TexTeller3-ONNX-dynamic',
  DEFAULT_PROVIDER: 'wasm',

  // Models
  PADDLE_VL_ID: 'PaddlePaddle/PaddleOCR-VL',
  PADDLE_VL_ONNX_REPO: 'USER_NAME/REPO_NAME', // Updated by user after upload
  PADDLE_VL_SERVER_URL: '', // e.g. http://localhost:8080/layout-parsing
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
    LLM: 'llm_init.onnx',
    LLM_WITH_PAST: 'llm_with_past.onnx',
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
  VLM_MEAN: [0.48145466, 0.4578275, 0.40821073],
  VLM_STD: [0.26862954, 0.26130258, 0.27577711],
  VLM_PATCH_SIZE: 14,
  VLM_MERGE_SIZE: 2,
  VLM_MIN_PIXELS: 28 * 28 * 130,
  VLM_MAX_PIXELS: 28 * 28 * 1280,
  VLM_TOKENS_PER_SECOND: 2,

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
