import { describe, it, expect } from 'vitest';

describe('Branch Protection Test', () => {
  it('should fail to test CI checks', () => {
    expect(true).toBe(false);
  });
});
