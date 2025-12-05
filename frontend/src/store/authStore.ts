import { create } from 'zustand';
import type { User } from '../types';
import { authService } from '../services/auth.service';

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  setUser: (user: User | null) => void;
  setToken: (token: string | null) => void;
  login: (user: User, token: string) => void;
  logout: () => void;
  initialize: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: null,
  isAuthenticated: false,
  isLoading: true,

  setUser: (user) => set({ user, isAuthenticated: !!user }),
  
  setToken: (token) => set({ token }),

  login: (user, token) => {
    authService.saveAuthData(token, user);
    set({ user, token, isAuthenticated: true });
  },

  logout: () => {
    authService.clearAuthData();
    set({ user: null, token: null, isAuthenticated: false });
  },

  initialize: () => {
    const token = authService.getStoredToken();
    const user = authService.getStoredUser();
    
    if (token && user) {
      set({ user, token, isAuthenticated: true, isLoading: false });
    } else {
      set({ isLoading: false });
    }
  },
}));
