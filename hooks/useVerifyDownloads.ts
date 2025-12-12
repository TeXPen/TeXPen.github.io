import { useAppContext } from '../contexts/AppContext';
import { MODEL_CONFIG } from '../services/inference/config';

export const useVerifyDownloads = () => {
  const {
    setCustomNotification,
    openDialog,
    closeDialog,
    provider,
    quantization,
    customModelId
  } = useAppContext();

  const verifyDownloads = async () => {
    const { modelLoader } = await import('../services/inference/ModelLoader');
    const { downloadManager } = await import('../services/downloader/DownloadManager');
    const { getSessionOptions } = await import('../services/inference/config');

    const toast = (msg: string | null | { message: string, progress?: number, isLoading?: boolean }) => {
      if (typeof msg === 'string') {
        setCustomNotification({ message: msg });
      } else {
        setCustomNotification(msg);
      }
    };

    const runVerification = async () => {
      try {
        toast('Verifying files...');
        // Get options for current custom ID or default
        const modelId = customModelId || MODEL_CONFIG.ID;
        const sessionOptions = getSessionOptions(provider); // Removed quantization arg

        const corrupted = await modelLoader.validateModelFiles(modelId, sessionOptions);

        if (corrupted.length > 0) {
          toast(null);
          openDialog({
            title: 'Missing or Corrupted Files',
            message: `Found ${corrupted.length} file(s) that need to be downloaded. Fix them now?`,
            confirmText: 'Download',
            isDangerous: false,
            onConfirm: async () => {
              closeDialog();
              toast('Repairing files...');
              for (const url of corrupted) {
                await downloadManager.deleteFromCache(url);
              }
              await modelLoader.preDownloadModels(modelId, sessionOptions, (status, progress) => {
                toast({ message: status, progress });
              });
              toast({ message: 'Repaired corrupted files!', isLoading: false });
              setTimeout(() => toast(null), 3000);
            }
          });
        } else {
          toast({ message: 'All files verified successfully.', isLoading: false });
          setTimeout(() => toast(null), 3000);
        }
      } catch (e) {
        console.error(e);
        toast({ message: 'Error detecting corruption. Check console.', isLoading: false });
        setTimeout(() => toast(null), 5000);
      }
    };

    openDialog({
      title: 'Verify Downloads',
      message: 'This will verify the integrity of downloaded model files and re-download any corrupted ones. Continue?',
      confirmText: 'Verify',
      isDangerous: false,
      onConfirm: () => {
        closeDialog();
        runVerification();
      }
    });
  };

  return { verifyDownloads };
};
