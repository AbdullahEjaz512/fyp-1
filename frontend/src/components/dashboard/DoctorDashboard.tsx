import { useQuery } from '@tanstack/react-query';
import { doctorService } from '../../services/file.service';
import { 
  Users, 
  Activity, 
  CheckCircle, 
  Clock, 
  Bell, 
  FileText,
  TrendingUp,
  AlertCircle,
  Calendar
} from 'lucide-react';
import './DoctorDashboard.css';

export const DoctorDashboard = () => {
  const { data: dashboard, isLoading, error } = useQuery({
    queryKey: ['doctor-dashboard'],
    queryFn: doctorService.getDashboard,
    retry: 1,
    refetchInterval: 60000, // Refresh every minute
  });

  console.log('DoctorDashboard - isLoading:', isLoading);
  console.log('DoctorDashboard - error:', error);
  console.log('DoctorDashboard - dashboard:', dashboard);

  if (isLoading) {
    return (
      <div className="dashboard-loading" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '400px', gap: '1rem' }}>
        <div className="loader" style={{ width: '50px', height: '50px', border: '4px solid #f3f3f3', borderTop: '4px solid #667eea', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></div>
        <p style={{ fontSize: '1.1rem', color: '#666' }}>Loading dashboard...</p>
      </div>
    );
  }

  if (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    console.error('DoctorDashboard error:', errorMessage);
    return (
      <div className="dashboard-error" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '400px', gap: '1rem', color: '#e53e3e', textAlign: 'center' }}>
        <AlertCircle size={48} color="#e53e3e" />
        <h3 style={{ fontSize: '1.5rem', margin: '0.5rem 0', color: '#333' }}>Error loading dashboard</h3>
        <p style={{ fontSize: '1rem', margin: '0.25rem 0' }}>{errorMessage}</p>
        <p className="error-hint" style={{ fontSize: '0.875rem', color: '#666', marginTop: '0.5rem' }}>Make sure you're logged in with a doctor account.</p>
      </div>
    );
  }

  if (!dashboard) {
    console.warn('DoctorDashboard - No dashboard data');
    return (
      <div className="dashboard-error" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '400px', gap: '1rem', color: '#e53e3e', textAlign: 'center' }}>
        <AlertCircle size={48} color="#e53e3e" />
        <h3 style={{ fontSize: '1.5rem', margin: '0.5rem 0', color: '#333' }}>No Dashboard Data</h3>
        <p style={{ fontSize: '1rem', margin: '0.25rem 0' }}>Unable to load dashboard information.</p>
      </div>
    );
  }

  return (
    <div className="doctor-dashboard">
      {/* Doctor Info Header */}
      <div className="dashboard-header">
        <div className="doctor-info">
          <div className="doctor-avatar">
            {dashboard.doctor_info?.name?.charAt(0) || 'D'}
          </div>
          <div>
            <h2 className="doctor-name">Dr. {dashboard.doctor_info?.name || 'Unknown'}</h2>
            <p className="doctor-spec">{dashboard.doctor_info?.specialization || 'Medical Professional'}</p>
          </div>
        </div>
      </div>

      {/* Statistics Cards */}
      <div className="stats-grid">
        <div className="stat-card primary">
          <div className="stat-icon-wrapper">
            <Users className="stat-icon" />
          </div>
          <div className="stat-content">
            <div className="stat-value">{dashboard.statistics.assigned_patients}</div>
            <div className="stat-label">Assigned Patients</div>
          </div>
          <div className="stat-trend">
            <TrendingUp size={16} />
            Active cases
          </div>
        </div>

        <div className="stat-card success">
          <div className="stat-icon-wrapper">
            <Activity className="stat-icon" />
          </div>
          <div className="stat-content">
            <div className="stat-value">{dashboard.statistics.total_analyses}</div>
            <div className="stat-label">Total Analyses</div>
          </div>
          <div className="stat-trend">
            <CheckCircle size={16} />
            All time
          </div>
        </div>

        <div className="stat-card info">
          <div className="stat-icon-wrapper">
            <Calendar className="stat-icon" />
          </div>
          <div className="stat-content">
            <div className="stat-value">{dashboard.statistics.analyses_this_month}</div>
            <div className="stat-label">This Month</div>
          </div>
          <div className="stat-trend">
            <Clock size={16} />
            Current period
          </div>
        </div>

        <div className="stat-card warning">
          <div className="stat-icon-wrapper">
            <Bell className="stat-icon" />
          </div>
          <div className="stat-content">
            <div className="stat-value">{dashboard.statistics.pending_assessments}</div>
            <div className="stat-label">Pending Assessments</div>
          </div>
          <div className="stat-trend">
            <AlertCircle size={16} />
            Requires attention
          </div>
        </div>
      </div>

      {/* Notifications */}
      {dashboard.notifications && dashboard.notifications.length > 0 && (
        <div className="notifications-section">
          <div className="section-header">
            <Bell size={24} />
            <h3>Notifications</h3>
            <span className="badge">{dashboard.notifications.length}</span>
          </div>
          <div className="notifications-list">
            {dashboard.notifications.map((notification, index) => (
              <div 
                key={index} 
                className={`notification-item priority-${notification.priority}`}
              >
                <div className="notification-icon">
                  {notification.type === 'pending_assessment' && <Clock size={20} />}
                  {notification.type === 'new_patients' && <Users size={20} />}
                </div>
                <div className="notification-content">
                  <p className="notification-message">{notification.message}</p>
                  <span className="notification-count">{notification.count} items</span>
                </div>
                <div className={`priority-badge priority-${notification.priority}`}>
                  {notification.priority}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Activities */}
      <div className="recent-activities-section">
        <div className="section-header">
          <Activity size={24} />
          <h3>Recent Activities</h3>
        </div>
        
        {dashboard.recent_activities && dashboard.recent_activities.length > 0 ? (
          <div className="activities-list">
            {dashboard.recent_activities.map((activity, index) => (
              <div key={index} className="activity-item">
                <div className="activity-icon">
                  <FileText size={20} />
                </div>
                <div className="activity-content">
                  <div className="activity-header">
                    <span className="activity-patient">Patient: {activity.patient_id}</span>
                    <span className="activity-date">
                      {new Date(activity.analysis_date).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="activity-filename">{activity.filename}</p>
                  <div className="activity-footer">
                    <span className="activity-tumor">{activity.tumor_type}</span>
                    {activity.has_assessment ? (
                      <span className="activity-status assessed">
                        <CheckCircle size={14} />
                        Assessed
                      </span>
                    ) : (
                      <span className="activity-status pending">
                        <Clock size={14} />
                        Pending
                      </span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <Activity size={48} />
            <p>No recent activities</p>
          </div>
        )}
      </div>

      {/* Assigned Patients List */}
      {dashboard.assigned_patients && dashboard.assigned_patients.length > 0 && (
        <div className="assigned-patients-section">
          <div className="section-header">
            <Users size={24} />
            <h3>Assigned Patients</h3>
            <span className="badge">{dashboard.assigned_patients.length}</span>
          </div>
          <div className="patients-grid">
            {dashboard.assigned_patients.map((patientId, index) => (
              <div key={index} className="patient-card">
                <div className="patient-avatar">
                  {patientId.slice(0, 2)}
                </div>
                <div className="patient-id">{patientId}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
