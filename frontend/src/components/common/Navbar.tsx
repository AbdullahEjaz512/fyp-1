import { Link, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../../store/authStore';
import { authService } from '../../services/auth.service';
import { Brain, Home, LayoutDashboard, Upload, FileText, LogOut, User } from 'lucide-react';
import './Navbar.css';

export const Navbar = () => {
  const navigate = useNavigate();
  const { user, isAuthenticated, logout } = useAuthStore();

  const handleLogout = async () => {
    try {
      await authService.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      logout();
      navigate('/');
    }
  };

  const displayName = user?.full_name || user?.username || user?.email;
  const isDoctorRole = user?.role && ['doctor', 'radiologist', 'oncologist'].includes(user.role);

  return (
    <nav className="navbar">
      <div className="nav-container">
        <Link to="/" className="nav-brand">
          <Brain size={32} />
          <div className="brand-text">
            <span className="brand-name">Seg-Mind</span>
            <span className="brand-tagline">Medical Imaging Analysis</span>
          </div>
        </Link>

        {isAuthenticated && (
          <div className="nav-links">
            <Link to="/" className="nav-link">
              <Home size={18} />
              Home
            </Link>
            {isDoctorRole && (
              <Link to="/dashboard" className="nav-link">
                <LayoutDashboard size={18} />
                Dashboard
              </Link>
            )}
            <Link to="/upload" className="nav-link">
              <Upload size={18} />
              Upload
            </Link>
            <Link to="/results" className="nav-link">
              <FileText size={18} />
              Results
            </Link>
          </div>
        )}

        <div className="nav-actions">
          {isAuthenticated && user ? (
            <div className="user-menu">
              <div className="user-info">
                <User size={18} />
                <div>
                  <div className="user-name">{displayName}</div>
                  {user.role === 'patient' && user.patient_id && (
                    <div className="user-role">ID: {user.patient_id}</div>
                  )}
                  {user.role !== 'patient' && (
                    <div className="user-role">{user.role}</div>
                  )}
                </div>
              </div>
              <button onClick={handleLogout} className="btn-secondary btn-sm">
                <LogOut size={16} />
                Logout
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </nav>
  );
};
