import { create } from 'zustand';

interface AppState {
  establishment: string;
  setEstablishment: (value: string) => void;
  alerts: number;
  setAlerts: (value: number) => void;
  theme: 'light' | 'dark';
  toggleTheme: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  establishment: 'Helios Energy HQ',
  alerts: 3,
  setEstablishment: (value) => set({ establishment: value }),
  setAlerts: (value) => set({ alerts: value }),
  theme: 'dark',
  toggleTheme: () =>
    set((state) => ({ theme: state.theme === 'dark' ? 'light' : 'dark' })),
}));
