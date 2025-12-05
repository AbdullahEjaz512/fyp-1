import { useAuthStore } from '../store/authStore';
import { DoctorDashboard } from '../components/dashboard/DoctorDashboard';
import { PatientDashboard } from '../components/dashboard/PatientDashboard';
import { LayoutDashboard, AlertCircle } from 'lucide-react';
import './DashboardPage.css';

export default function DashboardPage() {
  const user = useAuthStore((state) => state.user);
  
  console.log('DashboardPage - User:', user);
  
  if (!user) {
    return (
      <div className="dashboard-page">
        <div className="container">
          <div className="dashboard-error">
            <AlertCircle size={48} />
            <h3>Not Logged In</h3>
            <p>Please log in to view your dashboard.</p>
          </div>
        </div>
      </div>
    );
  }
  
  const isDoctorRole = user?.role && ['doctor', 'radiologist', 'oncologist'].includes(user.role);
  
  console.log('DashboardPage - isDoctorRole:', isDoctorRole);

  return (
    <div className="dashboard-page" style={{ minHeight: '100vh', background: '#f5f5f5', padding: '2rem' }}>
      <div className="container" style={{ maxWidth: '1200px', margin: '0 auto', background: 'white', padding: '2rem', borderRadius: '8px' }}>
        <div className="page-header" style={{ marginBottom: '2rem' }}>
          <h1 className="page-title" style={{ fontSize: '2rem', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            <LayoutDashboard size={32} />
            Dashboard
          </h1>
          <p className="page-subtitle" style={{ color: '#666', fontSize: '1rem' }}>
            {isDoctorRole 
              ? 'Your medical practice overview and patient management'
              : 'Overview of your medical files and analysis results'}
          </p>
        </div>

        <div style={{ background: '#fff', padding: '1rem', border: '2px solid #e0e0e0', borderRadius: '8px' }}>
          {isDoctorRole ? <DoctorDashboard /> : <PatientDashboard />}
        </div>
      </div>
    </div>
  );
}
