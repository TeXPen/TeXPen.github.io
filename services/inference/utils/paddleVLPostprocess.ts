
// Basic post-processing
// Real decoding happens in the loop, this is just for cleanup.

export function paddleVLPostprocess(text: string): string {
  // Fix common markdown issues or specific artifact cleanup
  let cleaned = text.trim();

  // Convert \( \) to $ $ if needed (Paddle might output LaTeX style)
  cleaned = cleaned.replace(/\\\((.*?)\\\)/g, '$$$1$$');
  cleaned = cleaned.replace(/\\\[(.*?)\\\]/g, '$$$$$1$$$$');

  return cleaned;
}
