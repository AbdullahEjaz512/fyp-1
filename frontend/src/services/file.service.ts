import api from './api';
import type { MRIFile, AnalysisResult, Doctor, DashboardData, DiscussionComment } from '../types';

export const fileService = {
  async uploadFile(file: File, patientId?: string): Promise<MRIFile> {
    const formData = new FormData();
    formData.append('file', file);
    if (patientId) {
      formData.append('patient_id', patientId);
    }

    const response = await api.post<MRIFile>('/api/v1/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async listFiles(): Promise<MRIFile[]> {
    const response = await api.get<MRIFile[]>('/api/v1/mri/files');
    return response.data;
  },

  async getFile(fileId: number): Promise<MRIFile> {
    const response = await api.get<MRIFile>(`/api/v1/mri/files/${fileId}`);
    return response.data;
  },

  async deleteFile(fileId: number): Promise<{ message: string }> {
    const response = await api.delete(`/api/v1/mri/files/${fileId}`);
    return response.data;
  },

  async preprocessFile(fileId: number, options?: {
    normalize?: string;
    denoise?: boolean;
    bias_correction?: boolean;
  }): Promise<any> {
    const response = await api.post(`/api/v1/mri/preprocess`, null, {
      params: { file_id: fileId, ...options },
    });
    return response.data;
  },

  async analyzeFile(fileId: number): Promise<AnalysisResult> {
    const response = await api.post<AnalysisResult>(`/api/v1/analyze`, null, {
      params: { file_id: fileId },
    });
    return response.data;
  },

  async getAnalysisResults(fileId: number): Promise<{
    file_id: number;
    patient_id: string;
    filename: string;
    total_analyses: number;
    analyses: AnalysisResult[];
  }> {
    const response = await api.get(`/api/v1/analyze/results/${fileId}`);
    return response.data;
  },

  async addDoctorAssessment(fileId: number, assessment: {
    doctor_interpretation?: string;
    clinical_diagnosis?: string;
    prescription?: string;
    treatment_plan?: string;
    follow_up_notes?: string;
    next_appointment?: string;
  }): Promise<any> {
    const response = await api.put(`/api/v1/analyze/results/${fileId}/assessment`, assessment);
    return response.data;
  },

  async downloadFile(fileId: number): Promise<Blob> {
    const response = await api.get(`/api/v1/mri/files/${fileId}/download`, {
      responseType: 'blob',
    });
    return response.data;
  },
};

export const doctorService = {
  async getAllDoctors(): Promise<{ doctors: Doctor[] }> {
    const response = await api.get<{ doctors: Doctor[] }>('/api/v1/doctors');
    return response.data;
  },

  async getDashboard(): Promise<DashboardData> {
    const response = await api.get<DashboardData>('/api/v1/doctors/dashboard');
    return response.data;
  },

  async listDoctors(): Promise<Doctor[]> {
    const response = await api.get<{ doctors: Doctor[] }>('/api/v1/doctors');
    return response.data.doctors;
  },

  async grantAccess(fileId: number, doctorIds: number[]): Promise<any> {
    const response = await api.post(`/api/v1/files/${fileId}/grant-access`, {
      doctor_ids: doctorIds,
    });
    return response.data;
  },

  async revokeAccess(fileId: number, doctorId: number): Promise<any> {
    const response = await api.delete(`/api/v1/files/${fileId}/revoke-access/${doctorId}`);
    return response.data;
  },

  async getFileAccess(fileId: number): Promise<{
    file_id: number;
    doctors_with_access: Array<{
      doctor_id: number;
      full_name: string;
      specialization: string;
      access_level: string;
      granted_date: string;
    }>;
  }> {
    const response = await api.get(`/api/v1/files/${fileId}/access`);
    return response.data;
  },

  async addDiscussion(fileId: number, comment: string, type: string = 'general'): Promise<any> {
    const response = await api.post(`/api/v1/cases/${fileId}/discussions`, {
      comment,
      type,
    });
    return response.data;
  },

  async getDiscussions(fileId: number): Promise<{
    file_id: number;
    patient_id: string;
    filename: string;
    total_comments: number;
    discussions: DiscussionComment[];
  }> {
    const response = await api.get(`/api/v1/cases/${fileId}/discussions`);
    return response.data;
  },
};

// Collaboration Service for multi-doctor features
export const collaborationService = {
  async shareCase(fileId: number, collaboratorEmail: string, message?: string): Promise<{
    success: boolean;
    message: string;
    collaboration_id: number;
    collaborator: {
      id: number;
      name: string;
      email: string;
      specialization: string;
    };
  }> {
    const formData = new FormData();
    formData.append('collaborator_email', collaboratorEmail);
    if (message) {
      formData.append('message', message);
    }
    const response = await api.post(`/api/v1/collaboration/share/${fileId}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  async getCollaborators(fileId: number): Promise<{
    file_id: number;
    collaborators: Array<{
      id: number;
      name: string;
      email: string;
      specialization: string;
      is_primary: boolean;
      has_analyzed: boolean;
      is_current_user: boolean;
    }>;
    total: number;
  }> {
    const response = await api.get(`/api/v1/collaboration/collaborators/${fileId}`);
    return response.data;
  },

  async removeCollaborator(fileId: number, doctorId: number): Promise<{ success: boolean; message: string }> {
    const response = await api.delete(`/api/v1/collaboration/remove/${fileId}/${doctorId}`);
    return response.data;
  },

  async getSharedWithMe(): Promise<{
    shared_cases: Array<{
      collaboration_id: number;
      file_id: number;
      filename: string;
      patient_id: string;
      shared_at: string;
      message: string;
      shared_by: {
        id: number;
        name: string;
        specialization: string;
      };
      has_analysis: boolean;
      classification_type: string;
    }>;
    total: number;
  }> {
    const response = await api.get('/api/v1/collaboration/shared-with-me');
    return response.data;
  },

  async addComment(fileId: number, comment: string, parentId?: number): Promise<{
    success: boolean;
    discussion_id: number;
    comment: {
      id: number;
      text: string;
      created_at: string;
      doctor: {
        id: number;
        name: string;
        specialization: string;
      };
      parent_id: number | null;
    };
  }> {
    const formData = new FormData();
    formData.append('comment', comment);
    if (parentId) {
      formData.append('parent_id', parentId.toString());
    }
    const response = await api.post(`/api/v1/discussion/${fileId}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  async getDiscussion(fileId: number): Promise<{
    file_id: number;
    comments: Array<{
      id: number;
      text: string;
      created_at: string;
      updated_at: string | null;
      doctor: {
        id: number;
        name: string;
        specialization: string;
        is_current_user: boolean;
      };
      parent_id: number | null;
    }>;
    total: number;
  }> {
    const response = await api.get(`/api/v1/discussion/${fileId}`);
    return response.data;
  },

  async updateComment(discussionId: number, comment: string): Promise<{ success: boolean; message: string }> {
    const formData = new FormData();
    formData.append('comment', comment);
    const response = await api.put(`/api/v1/discussion/${discussionId}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  },

  async deleteComment(discussionId: number): Promise<{ success: boolean; message: string }> {
    const response = await api.delete(`/api/v1/discussion/${discussionId}`);
    return response.data;
  },
};
