import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fileService } from '../../services/file.service';
import { useAuthStore } from '../../store/authStore';
import type { MRIFile } from '../../types';
import { 
  FileText, 
  Calendar, 
  HardDrive, 
  CheckCircle, 
  Clock, 
  Play, 
  Download,
  Trash2,
  Eye,
  Loader2,
  Users
} from 'lucide-react';
import { DoctorAccessModal } from './DoctorAccessModal';
import './FileList.css';

interface FileListProps {
  onAnalyze?: (fileId: number) => void;
  onViewResults?: (fileId: number) => void;
}

export const FileList = ({ onAnalyze, onViewResults }: FileListProps) => {
  const queryClient = useQueryClient();
  const user = useAuthStore((state) => state.user);
  const [deletingFileId, setDeletingFileId] = useState<number | null>(null);
  const [accessModalFileId, setAccessModalFileId] = useState<number | null>(null);
  const [accessModalFilename, setAccessModalFilename] = useState<string>('');

  const { data: files, isLoading, error } = useQuery({
    queryKey: ['files'],
    queryFn: fileService.listFiles,
    refetchInterval: (query) => {
      // Poll every 3 seconds if there are files being analyzed
      const hasAnalyzing = query.state.data?.some((f: MRIFile) => f.status === 'analyzing');
      return hasAnalyzing ? 3000 : false;
    },
  });

  const deleteMutation = useMutation({
    mutationFn: fileService.deleteFile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['files'] });
      setDeletingFileId(null);
    },
    onError: (error: any) => {
      console.error('Delete failed:', error);
      alert(`Failed to delete file: ${error?.response?.data?.detail || error.message || 'Unknown error'}`);
      setDeletingFileId(null);
    },
  });

  const downloadMutation = useMutation({
    mutationFn: fileService.downloadFile,
    onSuccess: (blob, fileId) => {
      const file = files?.find(f => f.file_id === fileId);
      if (file) {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = file.filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    },
  });

  const handleDelete = async (fileId: number) => {
    if (window.confirm('Are you sure you want to delete this file?')) {
      setDeletingFileId(fileId);
      try {
        await deleteMutation.mutateAsync(fileId);
      } catch (error) {
        // Error already handled in onError callback
        console.error('Delete mutation error:', error);
      }
    }
  };

  const handleDownload = (fileId: number) => {
    downloadMutation.mutate(fileId);
  };

  const handleGrantAccess = (fileId: number, filename: string) => {
    setAccessModalFileId(fileId);
    setAccessModalFilename(filename);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const getStatusBadge = (status: string) => {
    const statusMap: Record<string, { color: string; icon: any; label: string }> = {
      uploaded: { color: 'blue', icon: Clock, label: 'Uploaded' },
      preprocessed: { color: 'purple', icon: Clock, label: 'Preprocessed' },
      analyzing: { color: 'orange', icon: Loader2, label: 'Analyzing' },
      analyzed: { color: 'green', icon: CheckCircle, label: 'Analyzed' },
    };

    const statusInfo = statusMap[status] || statusMap.uploaded;
    const Icon = statusInfo.icon;

    return (
      <span className={`status-badge status-${statusInfo.color}`}>
        <Icon size={14} />
        {statusInfo.label}
      </span>
    );
  };

  if (isLoading) {
    return (
      <div className="file-list-loading">
        <Loader2 size={48} className="spinner" />
        <p>Loading files...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="file-list-error">
        <p>Error loading files: {error.message}</p>
      </div>
    );
  }

  if (!files || files.length === 0) {
    return (
      <div className="file-list-empty">
        <FileText size={64} className="empty-icon" />
        <h3>No files uploaded yet</h3>
        <p>Upload your first MRI scan to get started</p>
      </div>
    );
  }

  return (
    <>
      <div className="file-list">
        {files.map((file) => (
          <div key={file.file_id} className="file-card">
            <div className="file-card-header">
            <div className="file-info-main">
              <FileText size={24} className="file-icon-main" />
              <div>
                <h4 className="file-title">{file.filename}</h4>
                <p className="file-meta">
                  Patient ID: <strong>{file.patient_id}</strong>
                </p>
              </div>
            </div>
            {getStatusBadge(file.status)}
          </div>

          <div className="file-card-details">
            <div className="detail-item">
              <Calendar size={16} />
              <span>{formatDate(file.upload_date)}</span>
            </div>
            <div className="detail-item">
              <HardDrive size={16} />
              <span>{formatFileSize(file.size)}</span>
            </div>
            <div className="detail-item">
              <FileText size={16} />
              <span>{file.file_type}</span>
            </div>
            {file.analysis_count !== undefined && file.analysis_count > 0 && (
              <div className="detail-item">
                <CheckCircle size={16} />
                <span>{file.analysis_count} analysis</span>
              </div>
            )}
          </div>

          <div className="file-card-actions">
            {file.status === 'analyzed' && onViewResults && (
              <button
                onClick={() => onViewResults(file.file_id)}
                className="btn btn-primary btn-sm"
              >
                <Eye size={16} />
                View Results
              </button>
            )}

            {['uploaded', 'preprocessed'].includes(file.status) && onAnalyze && (
              <button
                onClick={() => onAnalyze(file.file_id)}
                className="btn btn-primary btn-sm"
              >
                <Play size={16} />
                Analyze
              </button>
            )}

            {user?.role === 'patient' && (
              <button
                onClick={() => handleGrantAccess(file.file_id, file.filename)}
                className="btn btn-secondary btn-sm"
              >
                <Users size={16} />
                Grant Access
              </button>
            )}

            <button
              onClick={() => handleDownload(file.file_id)}
              className="btn btn-secondary btn-sm"
              disabled={downloadMutation.isPending}
            >
              {downloadMutation.isPending ? (
                <Loader2 size={16} className="spinner" />
              ) : (
                <Download size={16} />
              )}
              Download
            </button>

            <button
              onClick={() => handleDelete(file.file_id)}
              className="btn btn-danger btn-sm"
              disabled={deletingFileId === file.file_id}
            >
              {deletingFileId === file.file_id ? (
                <Loader2 size={16} className="spinner" />
              ) : (
                <Trash2 size={16} />
              )}
              Delete
            </button>
          </div>
        </div>
        ))}
      </div>

      {accessModalFileId && (
        <DoctorAccessModal
          fileId={accessModalFileId}
          filename={accessModalFilename}
          onClose={() => {
            setAccessModalFileId(null);
            setAccessModalFilename('');
          }}
        />
      )}
    </>
  );
};
