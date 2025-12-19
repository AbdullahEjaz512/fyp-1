// Type definitions for Seg-Mind Medical Imaging Application

export interface User {
  user_id: string;
  email: string;
  username: string;
  full_name?: string;
  role: 'doctor' | 'radiologist' | 'oncologist' | 'patient' | 'admin';
  patient_id?: string;
  medical_license?: string;
  specialization?: string;
  institution?: string;
  department?: string;
  years_of_experience?: number;
  phone_number?: string;
  date_of_birth?: string;
  gender?: string;
  medical_record_number?: string;
  created_at?: string;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData {
  email: string;
  password: string;
  full_name: string;
  role: string;
  username?: string;
  medical_license?: string;
  specialization?: string;
  institution?: string;
  department?: string;
  years_of_experience?: number;
  phone_number?: string;
  date_of_birth?: string;
  gender?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface MRIFile {
  file_id: number;
  filename: string;
  safe_filename: string;
  path: string;
  size: number;
  file_type: string;
  uploaded_by: string;
  patient_id: string;
  upload_date: string;
  status: string;
  preprocessed: boolean;
  preprocessed_path?: string;
  metadata?: any;
  analysis_count?: number;
  has_analyzed?: boolean;
}

export interface SegmentationData {
  regions: {
    NCR: { volume_voxels: number; volume_mm3: number };
    ED: { volume_voxels: number; volume_mm3: number };
    ET: { volume_voxels: number; volume_mm3: number };
  };
  total_volume: {
    voxels: number;
    mm3: number;
  };
  metrics: {
    dice_score: number;
    hausdorff_distance: number;
  };
}

export interface ClassificationData {
  prediction: {
    tumor_type: string;
    confidence: number;
    who_grade: string;
    malignancy: string;
  };
  class_probabilities?: Record<string, number>;
}

export interface DoctorInfo {
  doctor_id: number;
  doctor_name: string;
  specialization: string;
  assessment_date: string;
}

export interface DoctorAssessment {
  interpretation?: string;
  diagnosis?: string;
  prescription?: string;
  treatment_plan?: string;
  follow_up_notes?: string;
  next_appointment?: string;
}

export interface EnsembleUncertainty {
  mean_confidence?: number;
  mean_entropy?: number;
  epistemic_uncertainty?: number;
  quality_flags?: {
    high_confidence?: boolean;
    low_uncertainty?: boolean;
    recommended_for_clinical_use?: boolean;
    requires_expert_review?: boolean;
  };
}

export interface EnsembleData {
  enabled: boolean;
  segmentation_uncertainty?: EnsembleUncertainty;
  classification_uncertainty?: EnsembleUncertainty;
}

export interface AnalysisResult {
  file_id: number;
  analysis_id: number;
  status: string;
  timestamp: string;
  patient_id: string;
  filename: string;
  segmentation: SegmentationData;
  classification: ClassificationData;
  summary: {
    diagnosis: string;
    confidence: number;
    tumor_volume_mm3: number;
    who_grade: string;
    malignancy: string;
  };
  total_time?: number;
  note?: string;
  doctor_info?: DoctorInfo;
  doctor_assessment?: DoctorAssessment;
  ensemble?: EnsembleData;
}

export interface Doctor {
  user_id: number;
  full_name: string;
  specialization: string;
  institution: string;
  department: string;
  years_of_experience: number;
  bio?: string;
}

export interface DashboardStats {
  assigned_patients: number;
  total_analyses: number;
  analyses_this_month: number;
  pending_assessments: number;
}

export interface Notification {
  type: string;
  message: string;
  count: number;
  priority: string;
}

export interface RecentActivity {
  file_id: number;
  patient_id: string;
  filename: string;
  analysis_date: string;
  has_assessment: boolean;
  tumor_type: string;
}

export interface DashboardData {
  doctor_info: {
    name: string;
    specialization: string;
    user_id: number;
  };
  statistics: DashboardStats;
  assigned_patients: string[];
  recent_activities: RecentActivity[];
  notifications: Notification[];
}

export interface DiscussionComment {
  discussion_id: string;
  user_id: number;
  doctor_name: string;
  specialization: string;
  comment: string;
  comment_type: string;
  timestamp: string;
  edited: boolean;
}
