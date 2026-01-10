// API Configuration
// Automatically uses Railway URL in production, localhost in development
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  // Auth
  login: `${API_BASE_URL}/api/v1/auth/login`,
  register: `${API_BASE_URL}/api/v1/auth/register`,
  
  // Files
  upload: `${API_BASE_URL}/api/v1/files/upload`,
  files: `${API_BASE_URL}/api/v1/files`,
  
  // Analysis
  segmentation: `${API_BASE_URL}/api/v1/analysis/segmentation`,
  classification: `${API_BASE_URL}/api/v1/analysis/classification`,
  
  // Reconstruction
  reconstruction: `${API_BASE_URL}/api/v1/reconstruction`,
  
  // Advanced
  advanced: `${API_BASE_URL}/api/v1/advanced`,
  
  // Assistant
  assistant: `${API_BASE_URL}/api/v1/assistant`,
};
