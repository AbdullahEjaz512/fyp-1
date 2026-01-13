import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../../store/authStore';
import { authService } from '../../services/auth.service';
import { LogOut, User, Settings } from 'lucide-react';
import { useState, useRef, useEffect } from 'react';
import './TopBar.css';

export const TopBar = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuthStore();
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

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

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  if (!user) return null;

  const initials = user.full_name
    ?.split(' ')
    .map(n => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2) || user.username?.slice(0, 2).toUpperCase() || 'U';

  const displayName = user.full_name || user.username || user.email;

  return (
    <div className="topbar">
      <div className="topbar-container">
        <div className="topbar-left">
          {/* Can add breadcrumbs or page title here if needed */}
        </div>

        <div className="topbar-right">
          <div className="user-menu" ref={dropdownRef}>
            <button 
              className="avatar-button"
              onClick={() => setShowDropdown(!showDropdown)}
              aria-label="User menu"
            >
              <div className="avatar-circle">
                {initials}
              </div>
              <div className="user-info-compact">
                <span className="user-name-compact">{displayName}</span>
                <span className="user-role-compact">{user.role}</span>
              </div>
            </button>

            {showDropdown && (
              <div className="dropdown-menu">
                <div className="dropdown-header">
                  <div className="dropdown-avatar">
                    {initials}
                  </div>
                  <div>
                    <div className="dropdown-name">{displayName}</div>
                    <div className="dropdown-email">{user.email}</div>
                    <div className="dropdown-role">{user.role}</div>
                  </div>
                </div>
                
                <div className="dropdown-divider"></div>
                
                <button className="dropdown-item" onClick={() => {
                  setShowDropdown(false);
                  navigate('/profile');
                }}>
                  <User size={16} />
                  <span>Profile Settings</span>
                </button>
                
                <button className="dropdown-item" onClick={() => {
                  setShowDropdown(false);
                  navigate('/settings');
                }}>
                  <Settings size={16} />
                  <span>Preferences</span>
                </button>
                
                <div className="dropdown-divider"></div>
                
                <button className="dropdown-item logout" onClick={handleLogout}>
                  <LogOut size={16} />
                  <span>Logout</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
