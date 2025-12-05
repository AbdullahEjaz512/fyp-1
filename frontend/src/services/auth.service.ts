import api from './api';
import type { LoginCredentials, RegisterData, AuthResponse, User } from '../types';

export const authService = {
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    const response = await api.post<AuthResponse>('/api/v1/auth/login', credentials);
    return response.data;
  },

  async register(data: RegisterData): Promise<{ message: string; user_id: string; role: string; patient_id?: string }> {
    const response = await api.post('/api/v1/auth/register', data);
    return response.data;
  },

  async logout(): Promise<void> {
    await api.post('/api/v1/auth/logout');
  },

  async getCurrentUser(): Promise<User> {
    const response = await api.get<User>('/api/v1/auth/me');
    return response.data;
  },

  async updateProfile(data: Partial<User>): Promise<{ message: string; user_id: string }> {
    const response = await api.put('/api/v1/auth/profile', data);
    return response.data;
  },

  saveAuthData(token: string, user: User): void {
    localStorage.setItem('authToken', token);
    localStorage.setItem('userData', JSON.stringify(user));
  },

  clearAuthData(): void {
    localStorage.removeItem('authToken');
    localStorage.removeItem('userData');
  },

  getStoredUser(): User | null {
    const userData = localStorage.getItem('userData');
    return userData ? JSON.parse(userData) : null;
  },

  getStoredToken(): string | null {
    return localStorage.getItem('authToken');
  },
};
