import { useState } from 'react';
import { useAuthStore } from '../../store/authStore';
import { authService } from '../../services/auth.service';
import { Brain, Mail, Lock, Loader2 } from 'lucide-react';
import './AuthForms.css';

interface LoginFormProps {
  onSuccess?: () => void;
  onSwitchToRegister?: () => void;
}

export const LoginForm = ({ onSuccess, onSwitchToRegister }: LoginFormProps) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  
  const login = useAuthStore((state) => state.login);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      const response = await authService.login({ email, password });
      login(response.user, response.access_token);
      onSuccess?.();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Login failed. Please check your credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="auth-form-container">
      <div className="auth-header">
        <Brain size={48} className="auth-icon" />
        <h2>Login to Seg-Mind</h2>
        <p>Access your medical imaging dashboard</p>
      </div>

      {error && <div className="error-message">{error}</div>}

      <form onSubmit={handleSubmit} className="auth-form">
        <div className="form-group">
          <label htmlFor="email">
            <Mail size={18} />
            Email
          </label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="doctor@hospital.com"
            required
            disabled={isLoading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="password">
            <Lock size={18} />
            Password
          </label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••"
            required
            disabled={isLoading}
          />
        </div>

        <button type="submit" className="btn-primary btn-block" disabled={isLoading}>
          {isLoading ? (
            <>
              <Loader2 size={18} className="spinner" />
              Logging in...
            </>
          ) : (
            'Login'
          )}
        </button>

        <p className="form-footer">
          Don't have an account?{' '}
          <button type="button" onClick={onSwitchToRegister} className="link-button">
            Sign up
          </button>
        </p>
      </form>
    </div>
  );
};
