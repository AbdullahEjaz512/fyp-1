import { useState } from 'react';
import { useAuthStore } from '../store/authStore';
import { LoginForm } from '../components/auth/LoginForm';
import { RegisterForm } from '../components/auth/RegisterForm';
import { Brain, CheckCircle, Clock, Shield, Zap } from 'lucide-react';
import './HomePage.css';

export default function HomePage() {
  const [showLogin, setShowLogin] = useState(false);
  const [showRegister, setShowRegister] = useState(false);
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  const user = useAuthStore((state) => state.user);

  const handleAuthSuccess = () => {
    setShowLogin(false);
    setShowRegister(false);
  };

  if (isAuthenticated && user) {
    return (
      <div className="home-page">
        <section className="hero">
          <div className="hero-content">
            <Brain size={64} className="hero-icon" />
            <h1 className="hero-title">Welcome back, {user.full_name || user.username}!</h1>
            <p className="hero-subtitle">
              {user.role === 'patient' 
                ? 'Manage your MRI scans and view analysis results'
                : 'Access your doctor dashboard and patient cases'}
            </p>
            <div className="hero-stats">
              <div className="stat-card">
                <CheckCircle className="stat-icon" />
                <div className="stat-value">96.6%</div>
                <div className="stat-label">Accuracy</div>
              </div>
              <div className="stat-card">
                <Clock className="stat-icon" />
                <div className="stat-value">&lt;10s</div>
                <div className="stat-label">Processing</div>
              </div>
              <div className="stat-card">
                <Shield className="stat-icon" />
                <div className="stat-value">HIPAA</div>
                <div className="stat-label">Compliant</div>
              </div>
            </div>
          </div>
        </section>

        <section className="features-section">
          <div className="container">
            <h2 className="section-title">Platform Features</h2>
            <div className="features-grid">
              <div className="feature-card">
                <Brain className="feature-icon" />
                <h3>3D Tumor Segmentation</h3>
                <p>Automated detection and volumetric analysis of tumor regions</p>
              </div>
              <div className="feature-card">
                <Zap className="feature-icon" />
                <h3>AI Classification</h3>
                <p>Identify tumor types with high confidence scoring</p>
              </div>
              <div className="feature-card">
                <Shield className="feature-icon" />
                <h3>Secure & Private</h3>
                <p>HIPAA-compliant data storage and transmission</p>
              </div>
            </div>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="home-page">
      <section className="hero">
        <div className="hero-content">
          <Brain size={64} className="hero-icon" />
          <h1 className="hero-title">Brain Tumor Analysis System</h1>
          <p className="hero-subtitle">
            Advanced Medical Imaging Platform powered by Deep Learning
          </p>
          <div className="hero-buttons">
            <button className="btn btn-primary btn-lg" onClick={() => setShowLogin(true)}>
              Login
            </button>
            <button className="btn btn-secondary btn-lg" onClick={() => setShowRegister(true)}>
              Sign Up
            </button>
          </div>
          <div className="hero-stats">
            <div className="stat-card">
              <CheckCircle className="stat-icon" />
              <div className="stat-value">96.6%</div>
              <div className="stat-label">Classification Accuracy</div>
            </div>
            <div className="stat-card">
              <Clock className="stat-icon" />
              <div className="stat-value">&lt;10s</div>
              <div className="stat-label">Processing Time</div>
            </div>
            <div className="stat-card">
              <Brain className="stat-icon" />
              <div className="stat-value">4</div>
              <div className="stat-label">Tumor Types</div>
            </div>
            <div className="stat-card">
              <Shield className="stat-icon" />
              <div className="stat-value">HIPAA</div>
              <div className="stat-label">Compliant</div>
            </div>
          </div>
        </div>
      </section>

      <section className="features-section">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Clinical Features</h2>
            <p className="section-subtitle">Comprehensive diagnostic tools for brain tumor analysis</p>
          </div>
          <div className="features-grid">
            <div className="feature-card">
              <Brain className="feature-icon primary" />
              <h3>3D Tumor Segmentation</h3>
              <p>Automated segmentation of tumor regions including necrotic core, edema, and enhancing tumor using validated algorithms</p>
              <ul className="feature-list">
                <li>Multi-region delineation</li>
                <li>Volumetric analysis</li>
                <li>3D visualization</li>
              </ul>
            </div>
            <div className="feature-card">
              <Zap className="feature-icon success" />
              <h3>Tumor Classification</h3>
              <p>Diagnostic classification system identifying tumor pathology and grade based on imaging characteristics</p>
              <ul className="feature-list">
                <li>Multiple tumor types</li>
                <li>Grade assessment</li>
                <li>Confidence scoring</li>
              </ul>
            </div>
            <div className="feature-card">
              <Shield className="feature-icon warning" />
              <h3>Data Security</h3>
              <p>HIPAA-compliant platform with secure authentication and role-based access control</p>
              <ul className="feature-list">
                <li>Secure authentication</li>
                <li>Role management</li>
                <li>Audit logging</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {showLogin && (
        <div className="modal-overlay" onClick={() => setShowLogin(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowLogin(false)}>×</button>
            <LoginForm 
              onSuccess={handleAuthSuccess}
              onSwitchToRegister={() => {
                setShowLogin(false);
                setShowRegister(true);
              }}
            />
          </div>
        </div>
      )}

      {showRegister && (
        <div className="modal-overlay" onClick={() => setShowRegister(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setShowRegister(false)}>×</button>
            <RegisterForm 
              onSuccess={handleAuthSuccess}
              onSwitchToLogin={() => {
                setShowRegister(false);
                setShowLogin(true);
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
