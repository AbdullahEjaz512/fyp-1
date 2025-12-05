import { useQuery } from '@tanstack/react-query';
import { fileService } from '../../services/file.service';
import { 
  FileText, 
  Calendar, 
  Activity, 
  TrendingUp,
  CheckCircle,
  Clock,
  AlertCircle
} from 'lucide-react';
import './PatientDashboard.css';

export const PatientDashboard = () => {
  const { data: files, isLoading, error } = useQuery({
    queryKey: ['files'],
    queryFn: fileService.listFiles,
  });

  if (isLoading) {
    return (
      <div className="dashboard-loading">
        <div className="loader"></div>
        <p>Loading dashboard...</p>
      </div>
    );
  }

  if (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return (
      <div className="dashboard-error">
        <AlertCircle size={48} />
        <h3>Error loading dashboard</h3>
        <p>{errorMessage}</p>
      </div>
    );
  }

  const totalFiles = files?.length || 0;
  const analyzedFiles = files?.filter(f => f.status === 'analyzed').length || 0;
  const pendingFiles = files?.filter(f => f.status !== 'analyzed').length || 0;
  const recentFiles = files?.slice(0, 5) || [];

  return (
    <div className="patient-dashboard">
      <div className="dashboard-welcome">
        <h2>Your Medical Files Dashboard</h2>
        <p>Track your MRI scans and analysis results</p>
      </div>

      {/* Statistics */}
      <div className="stats-grid">
        <div className="stat-card primary">
          <div className="stat-icon-wrapper">
            <FileText className="stat-icon" />
          </div>
          <div className="stat-content">
            <div className="stat-value">{totalFiles}</div>
            <div className="stat-label">Total Files</div>
          </div>
          <div className="stat-trend">
            <TrendingUp size={16} />
            All uploads
          </div>
        </div>

        <div className="stat-card success">
          <div className="stat-icon-wrapper">
            <CheckCircle className="stat-icon" />
          </div>
          <div className="stat-content">
            <div className="stat-value">{analyzedFiles}</div>
            <div className="stat-label">Analyzed</div>
          </div>
          <div className="stat-trend">
            <Activity size={16} />
            Completed
          </div>
        </div>

        <div className="stat-card warning">
          <div className="stat-icon-wrapper">
            <Clock className="stat-icon" />
          </div>
          <div className="stat-content">
            <div className="stat-value">{pendingFiles}</div>
            <div className="stat-label">Pending</div>
          </div>
          <div className="stat-trend">
            <Clock size={16} />
            Awaiting analysis
          </div>
        </div>
      </div>

      {/* Recent Files */}
      <div className="recent-files-section">
        <div className="section-header">
          <Activity size={24} />
          <h3>Recent Uploads</h3>
        </div>
        
        {recentFiles.length > 0 ? (
          <div className="recent-files-list">
            {recentFiles.map((file) => (
              <div key={file.file_id} className="recent-file-item">
                <div className="file-icon">
                  <FileText size={20} />
                </div>
                <div className="file-info">
                  <div className="file-name">{file.filename}</div>
                  <div className="file-date">
                    <Calendar size={14} />
                    {new Date(file.upload_date).toLocaleDateString()}
                  </div>
                </div>
                <span className={`status-badge status-${file.status}`}>
                  {file.status}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <FileText size={48} />
            <p>No files uploaded yet</p>
            <p className="empty-hint">Upload your first MRI scan to get started</p>
          </div>
        )}
      </div>
    </div>
  );
};
