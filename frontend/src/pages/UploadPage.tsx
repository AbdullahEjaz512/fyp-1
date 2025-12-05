import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { fileService, collaborationService } from '../services/file.service';
import { FileUpload, UploadedFileItem } from '../components/upload/FileUpload';
import { FileList } from '../components/upload/FileList';
import { Upload, AlertCircle, CheckCircle, Loader2, User, Users, FileText, Calendar } from 'lucide-react';
import './UploadPage.css';

interface FileWithProgress {
  file: File;
  progress: number;
  isUploading: boolean;
  error?: string;
  fileId?: number;
}

export default function UploadPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const user = useAuthStore((state) => state.user);
  const [selectedFiles, setSelectedFiles] = useState<FileWithProgress[]>([]);
  const [patientId, setPatientId] = useState('');

  const isDoctorRole = user?.role && ['doctor', 'radiologist', 'oncologist'].includes(user.role);

  // Query for cases shared with the current doctor
  const { data: sharedCases } = useQuery({
    queryKey: ['shared-cases'],
    queryFn: () => collaborationService.getSharedWithMe(),
    enabled: isDoctorRole,
  });

  const uploadMutation = useMutation({
    mutationFn: async ({ file, patientId }: { file: File; patientId?: string }) => {
      return await fileService.uploadFile(file, patientId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['files'] });
    },
  });

  const analyzeMutation = useMutation({
    mutationFn: fileService.analyzeFile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['files'] });
    },
  });

  const handleFilesSelected = (files: File[]) => {
    const newFiles: FileWithProgress[] = files.map(file => ({
      file,
      progress: 0,
      isUploading: false,
    }));
    setSelectedFiles(prev => [...prev, ...newFiles]);
  };

  const handleRemoveFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleUploadAll = async () => {
    if (isDoctorRole && !patientId.trim()) {
      alert('Please enter a Patient ID');
      return;
    }

    for (let i = 0; i < selectedFiles.length; i++) {
      const fileWithProgress = selectedFiles[i];
      
      if (fileWithProgress.isUploading || fileWithProgress.fileId) continue;

      // Mark as uploading
      setSelectedFiles(prev => {
        const updated = [...prev];
        updated[i] = { ...updated[i], isUploading: true, progress: 0 };
        return updated;
      });

      try {
        // Simulate progress
        const progressInterval = setInterval(() => {
          setSelectedFiles(prev => {
            const updated = [...prev];
            if (updated[i] && updated[i].progress < 90) {
              updated[i] = { ...updated[i], progress: updated[i].progress + 10 };
            }
            return updated;
          });
        }, 200);

        const result = await uploadMutation.mutateAsync({
          file: fileWithProgress.file,
          patientId: isDoctorRole ? patientId : undefined,
        });

        clearInterval(progressInterval);

        // Complete upload
        setSelectedFiles(prev => {
          const updated = [...prev];
          updated[i] = { 
            ...updated[i], 
            progress: 100, 
            isUploading: false,
            fileId: result.file_id 
          };
          return updated;
        });

      } catch (error: any) {
        setSelectedFiles(prev => {
          const updated = [...prev];
          updated[i] = { 
            ...updated[i], 
            isUploading: false, 
            error: error.response?.data?.detail || 'Upload failed' 
          };
          return updated;
        });
      }
    }
  };

  const handleAnalyze = async (fileId: number) => {
    try {
      await analyzeMutation.mutateAsync(fileId);
      // Navigate directly to results page after successful analysis
      navigate(`/results?fileId=${fileId}`);
    } catch (error: any) {
      alert(error.response?.data?.detail || 'Analysis failed');
    }
  };

  const handleViewResults = (fileId: number) => {
    navigate(`/results?fileId=${fileId}`);
  };

  const hasFilesToUpload = selectedFiles.length > 0;
  const allFilesUploaded = selectedFiles.every(f => f.fileId !== undefined);
  const isUploading = selectedFiles.some(f => f.isUploading);

  return (
    <div className="upload-page">
      <div className="container">
        <div className="page-header">
          <div>
            <h1 className="page-title">
              <Upload size={32} />
              Upload MRI Scans
            </h1>
            <p className="page-subtitle">
              {isDoctorRole 
                ? 'Upload patient MRI scans for AI-powered analysis'
                : 'Upload your MRI scans and get professional analysis'}
            </p>
          </div>
        </div>

        {/* Patient ID Input (for doctors) */}
        {isDoctorRole && (
          <div className="patient-selection-card">
            <div className="card-header-row">
              <User size={24} className="section-icon" />
              <h3>Patient Information</h3>
            </div>
            <div className="patient-input-group">
              <div className="form-group">
                <label htmlFor="patientId">
                  Patient ID <span className="required">*</span>
                </label>
                <input
                  id="patientId"
                  type="text"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  placeholder="e.g., PT-2025-00001"
                  className="input-field"
                />
                <p className="input-hint">
                  Enter the patient's unique identifier
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Upload Section */}
        <div className="upload-section-card">
          <div className="card-header-row">
            <h3>Upload Files</h3>
            {hasFilesToUpload && (
              <span className="file-count-badge">
                {selectedFiles.length} file{selectedFiles.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>

          <FileUpload onFilesSelected={handleFilesSelected} />

          {hasFilesToUpload && (
            <div className="selected-files-section">
              <h4>Selected Files</h4>
              <div className="selected-files-list">
                {selectedFiles.map((fileWithProgress, index) => (
                  <UploadedFileItem
                    key={index}
                    file={fileWithProgress.file}
                    uploadProgress={fileWithProgress.progress}
                    isUploading={fileWithProgress.isUploading}
                    error={fileWithProgress.error}
                    onRemove={() => handleRemoveFile(index)}
                  />
                ))}
              </div>

              {!allFilesUploaded && (
                <div className="upload-actions">
                  <button
                    onClick={handleUploadAll}
                    disabled={isUploading || (isDoctorRole && !patientId.trim())}
                    className="btn btn-primary btn-lg"
                  >
                    {isUploading ? (
                      <>
                        <Loader2 size={20} className="spinner" />
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Upload size={20} />
                        Upload All Files
                      </>
                    )}
                  </button>
                  {isDoctorRole && !patientId.trim() && (
                    <p className="upload-hint">
                      <AlertCircle size={16} />
                      Please enter Patient ID before uploading
                    </p>
                  )}
                </div>
              )}

              {allFilesUploaded && (
                <div className="upload-success">
                  <CheckCircle size={24} />
                  <span>All files uploaded successfully!</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Shared Cases Section (for doctors) */}
        {isDoctorRole && sharedCases && sharedCases.shared_cases.length > 0 && (
          <div className="shared-cases-section">
            <h2 className="section-title">
              <Users size={24} />
              Cases Shared With You ({sharedCases.total})
            </h2>
            <div className="shared-cases-grid">
              {sharedCases.shared_cases.map((sharedCase) => (
                <div key={sharedCase.collaboration_id} className="shared-case-card">
                  <div className="shared-case-header">
                    <FileText size={20} className="file-icon" />
                    <div className="shared-case-info">
                      <h4>{sharedCase.filename}</h4>
                      <span className="patient-id">Patient: {sharedCase.patient_id}</span>
                    </div>
                  </div>
                  <div className="shared-case-meta">
                    <div className="shared-by">
                      <span className="label">Shared by:</span>
                      <span className="value">Dr. {sharedCase.shared_by.name}</span>
                      {sharedCase.shared_by.specialization && (
                        <span className="specialization">({sharedCase.shared_by.specialization})</span>
                      )}
                    </div>
                    <div className="shared-date">
                      <Calendar size={14} />
                      {new Date(sharedCase.shared_at).toLocaleDateString()}
                    </div>
                  </div>
                  {sharedCase.message && (
                    <div className="shared-message">
                      <em>"{sharedCase.message}"</em>
                    </div>
                  )}
                  {sharedCase.has_analysis && (
                    <div className="analysis-badge">
                      <CheckCircle size={14} />
                      {sharedCase.classification_type || 'Analyzed'}
                    </div>
                  )}
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={() => navigate(`/results?fileId=${sharedCase.file_id}`)}
                  >
                    View Case
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* File History */}
        <div className="file-history-section">
          <h2 className="section-title">Your Files</h2>
          <FileList 
            onAnalyze={handleAnalyze}
            onViewResults={handleViewResults}
          />
        </div>
      </div>
    </div>
  );
}
