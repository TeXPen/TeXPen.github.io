import { PerformanceProfile } from '../services/inference/types';

export async function isWebGPUAvailable(): Promise<boolean> {
  if (!navigator.gpu) {
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

export interface DeviceCapabilities {
  hasGPU: boolean;
  memoryGB?: number;
}

export async function getDeviceCapabilities(): Promise<DeviceCapabilities> {
  const hasGPU = await isWebGPUAvailable();

  // navigator.deviceMemory returns approximate RAM in GB (can be 0.25, 0.5, 1, 2, 4, 8)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const memoryGB = (navigator as any).deviceMemory;

  return {
    hasGPU,
    memoryGB
  };
}

export async function getDefaultProfile(): Promise<PerformanceProfile> {
  const capabilities = await getDeviceCapabilities();

  console.log('[env] Device Capabilities:', capabilities);

  if (capabilities.hasGPU) {
    if (capabilities.memoryGB === undefined || capabilities.memoryGB >= 4) {
      return 'high_quality';
    }
  }

  // Fallback for CPU-only or low memory devices
  return 'balanced';
}
