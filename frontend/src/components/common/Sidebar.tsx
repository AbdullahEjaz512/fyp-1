import { Link, useLocation } from 'react-router-dom';
import { useAuthStore } from '../../store/authStore';
import {
  Home,
  LayoutDashboard,
  Upload,
  FileText,
  Eye,
  Box,
  TrendingUp,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';
import { useUIStore } from '../../store/uiStore';
import './Sidebar.css';

export const Sidebar = () => {
  const location = useLocation();
  const { user } = useAuthStore();
  const { isSidebarCollapsed: collapsed, toggleSidebar } = useUIStore();

  const isDoctorRole = user?.role && ['doctor', 'radiologist', 'oncologist'].includes(user.role);

  const doctorNavItems = [
    { path: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/upload', icon: Upload, label: 'Upload New Scan' },
    { path: '/results', icon: FileText, label: 'Results' },
    { path: '/visualization', icon: Eye, label: '2D Visualization' },
    { path: '/reconstruction', icon: Box, label: '3D Reconstruction' },
    { path: '/growth-prediction', icon: TrendingUp, label: 'Growth Prediction' },
  ];

  const patientNavItems = [
    { path: '/dashboard', icon: Home, label: 'Home' },
    { path: '/upload', icon: Upload, label: 'Upload Scan' },
    { path: '/results', icon: FileText, label: 'My Results' },
  ];

  const navItems = isDoctorRole ? doctorNavItems : patientNavItems;

  const isActive = (path: string) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        {!collapsed && (
          <div className="sidebar-brand">
            <div className="brand-icon">🧠</div>
            <div className="brand-info">
              <span className="brand-name">Seg-Mind</span>
              <span className="brand-tagline">Medical Imaging</span>
            </div>
          </div>
        )}
        {collapsed && <div className="brand-icon-only">🧠</div>}

        <button
          className="collapse-btn"
          onClick={toggleSidebar}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
        </button>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <Link
              key={item.path}
              to={item.path}
              className={`sidebar-link ${isActive(item.path) ? 'active' : ''}`}
              title={collapsed ? item.label : ''}
            >
              <Icon size={20} className="sidebar-icon" />
              {!collapsed && <span className="sidebar-label">{item.label}</span>}
            </Link>
          );
        })}
      </nav>

      {user && !collapsed && (
        <div className="sidebar-footer">
          <div className="sidebar-user">
            <div className="user-avatar">
              {user.full_name?.split(' ').map(n => n[0]).join('').toUpperCase() || 'U'}
            </div>
            <div className="user-details">
              <div className="user-name">{user.full_name || user.username}</div>
              <div className="user-role">{user.role}</div>
            </div>
          </div>
        </div>
      )}

      {user && collapsed && (
        <div className="sidebar-footer">
          <div className="user-avatar-only">
            {user.full_name?.split(' ').map(n => n[0]).join('').toUpperCase() || 'U'}
          </div>
        </div>
      )}
    </aside>
  );
};
