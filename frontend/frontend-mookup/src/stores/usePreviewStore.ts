import { create } from 'zustand';

interface PreviewState {
  current: string | null;
  open: (id: string) => void;
  close: () => void;
}

export const usePreviewStore = create<PreviewState>((set) => ({
  current: null,
  open: (id) => set({ current: id }),
  close: () => set({ current: null }),
}));
