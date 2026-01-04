export const MODEL_CONFIG = {
  ID: 'Ji-Ha/TexTeller3-ONNX-dynamic',
  DEFAULT_PROVIDER: 'wasm',

  // Model Specs
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
    'encoder_model.onnx': 'ae5e60838e4498412a02561e9f9ef176f30207a8405e002c4105f93c421efabc',
    'encoder_model_fp16.onnx': 'f78538bdee77c4eccb8e0b54f9058b087f08167b7500d893b308266323b3a23b',
    'encoder_model_int8.onnx': 'f6757c000c4cda7f6286a999193446b6618735e97cf2b46b8e0d62e51cfc71a8',
    'encoder_model_int4.onnx': 'cd6b555997aeedf9926a5397c3741a9947143d4d27c6c17b6f21dcaa359b40de',
    'decoder_model_merged.onnx': '4fc84dbc62be9a9115ffdf51de055ac39b12dacc2a5c70d921252d17ff745ad3',
    'decoder_model_merged_int8.onnx': '2590040b23aa50e76af8ad6c211edd34c1ed47439e31fd7b4ff935051b2efcac',
    'decoder_model_merged_int4.onnx': 'db0c9bfe85401afa7f9468eab5e00dd1b9f19b11f425530123fcac8c47fa9871',
  },
} as const;
