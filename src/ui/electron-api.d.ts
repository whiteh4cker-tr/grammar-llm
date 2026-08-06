import type { IpcApi } from '../electron/ipc-types';

declare global {
  interface Window {
    api: IpcApi;
  }
}

export {};
