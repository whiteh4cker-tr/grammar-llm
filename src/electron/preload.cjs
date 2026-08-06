const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  correct: (text) => ipcRenderer.invoke('text:correct', { text }),
  applySuggestion: (args) => ipcRenderer.invoke('suggestion:apply', args),
  applyMany: (args) => ipcRenderer.invoke('suggestion:applyMany', args),
  modelStatus: () => ipcRenderer.invoke('model:status'),
  downloadModel: (args) => ipcRenderer.invoke('model:download', args),
  cancelDownload: () => ipcRenderer.invoke('model:cancel-download'),
  onDownloadProgress: (cb) => {
    const listener = (_event, progress) => cb(progress);
    ipcRenderer.on('model:download-progress', listener);
    return () => ipcRenderer.removeListener('model:download-progress', listener);
  },
});
