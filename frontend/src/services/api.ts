import axios from 'axios';

// Use environment variable for API URL (set in Vercel), fallback to localhost for dev
const ENV_API_URL = import.meta.env.VITE_API_URL;
const IS_PROD = import.meta.env.PROD;
const FALLBACK_URL = IS_PROD 
  ? 'https://fyp-1-production.up.railway.app' 
  : 'http://127.0.0.1:8000';

export const API_BASE_URL = ENV_API_URL || FALLBACK_URL;

console.log('🔌 API Configuration:', { 
  envUrl: ENV_API_URL, 
  isProd: IS_PROD, 
  finalUrl: API_BASE_URL 
});

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('authToken');
      localStorage.removeItem('userData');
      window.location.href = '/';
    }
    return Promise.reject(error);
  }
);

export default api;
