import { useState } from 'react';
import { useAuthStore } from '../../store/authStore';
import { authService } from '../../services/auth.service';
import type { RegisterData } from '../../types';
import { Brain, Mail, User, Lock, Loader2 } from 'lucide-react';
import './AuthForms.css';

interface RegisterFormProps {
  onSuccess?: () => void;
  onSwitchToLogin?: () => void;
}

export const RegisterForm = ({ onSuccess, onSwitchToLogin }: RegisterFormProps) => {
  const [formData, setFormData] = useState<RegisterData>({
    email: '',
    password: '',
    full_name: '',
    role: '',
    medical_license: '',
    specialization: '',
    institution: '',
    department: '',
    years_of_experience: 0,
    phone_number: '',
    date_of_birth: '',
    gender: '',
  });
  const [confirmPassword, setConfirmPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const login = useAuthStore((state) => state.login);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'years_of_experience' ? parseInt(value) || 0 : value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (formData.password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (formData.password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    const isDoctorRole = ['doctor', 'radiologist', 'oncologist'].includes(formData.role);
    if (isDoctorRole && (!formData.specialization || !formData.medical_license || !formData.institution)) {
      setError('Please fill in all required doctor fields');
      return;
    }

    setIsLoading(true);

    try {
      await authService.register(formData);
      setSuccess('Registration successful! Logging you in...');
      
      // Auto-login after registration
      setTimeout(async () => {
        const loginResponse = await authService.login({
          email: formData.email,
          password: formData.password,
        });
        login(loginResponse.user, loginResponse.access_token);
        onSuccess?.();
      }, 1000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Registration failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const isDoctorRole = ['doctor', 'radiologist', 'oncologist'].includes(formData.role);

  return (
    <div className="auth-form-container">
      <div className="auth-header">
        <Brain size={48} className="auth-icon" />
        <h2>Create Account</h2>
        <p>Join Seg-Mind Medical Platform</p>
      </div>

      {error && <div className="error-message">{error}</div>}
      {success && <div className="success-message">{success}</div>}

      <form onSubmit={handleSubmit} className="auth-form">
        <div className="form-group">
          <label htmlFor="full_name">
            <User size={18} />
            Full Name *
          </label>
          <input
            id="full_name"
            name="full_name"
            type="text"
            value={formData.full_name}
            onChange={handleChange}
            placeholder="Dr. John Doe"
            required
            disabled={isLoading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="email">
            <Mail size={18} />
            Email *
          </label>
          <input
            id="email"
            name="email"
            type="email"
            value={formData.email}
            onChange={handleChange}
            placeholder="doctor@hospital.com"
            required
            disabled={isLoading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="role">Role *</label>
          <select
            id="role"
            name="role"
            value={formData.role}
            onChange={handleChange}
            required
            disabled={isLoading}
          >
            <option value="">Select Role</option>
            <option value="doctor">Doctor</option>
            <option value="radiologist">Radiologist</option>
            <option value="oncologist">Oncologist</option>
            <option value="patient">Patient</option>
          </select>
        </div>

        {isDoctorRole && (
          <>
            <div className="form-group">
              <label htmlFor="specialization">Specialization *</label>
              <input
                id="specialization"
                name="specialization"
                type="text"
                value={formData.specialization}
                onChange={handleChange}
                placeholder="e.g., Neuroradiology, Neurosurgery"
                required
                disabled={isLoading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="medical_license">Medical License *</label>
              <input
                id="medical_license"
                name="medical_license"
                type="text"
                value={formData.medical_license}
                onChange={handleChange}
                placeholder="MD-12345"
                required
                disabled={isLoading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="institution">Institution *</label>
              <input
                id="institution"
                name="institution"
                type="text"
                value={formData.institution}
                onChange={handleChange}
                placeholder="Hospital/Clinic Name"
                required
                disabled={isLoading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="department">Department</label>
              <input
                id="department"
                name="department"
                type="text"
                value={formData.department}
                onChange={handleChange}
                placeholder="e.g., Radiology, Oncology"
                disabled={isLoading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="years_of_experience">Years of Experience</label>
              <input
                id="years_of_experience"
                name="years_of_experience"
                type="number"
                value={formData.years_of_experience}
                onChange={handleChange}
                min="0"
                max="70"
                disabled={isLoading}
              />
            </div>
          </>
        )}

        {formData.role === 'patient' && (
          <div className="form-group">
            <label htmlFor="date_of_birth">Date of Birth</label>
            <input
              id="date_of_birth"
              name="date_of_birth"
              type="date"
              value={formData.date_of_birth}
              onChange={handleChange}
              disabled={isLoading}
            />
          </div>
        )}

        <div className="form-group">
          <label htmlFor="phone_number">Phone Number</label>
          <input
            id="phone_number"
            name="phone_number"
            type="tel"
            value={formData.phone_number}
            onChange={handleChange}
            placeholder="+1-555-0123"
            disabled={isLoading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="password">
            <Lock size={18} />
            Password *
          </label>
          <input
            id="password"
            name="password"
            type="password"
            value={formData.password}
            onChange={handleChange}
            placeholder="••••••••"
            required
            minLength={8}
            disabled={isLoading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="confirmPassword">
            <Lock size={18} />
            Confirm Password *
          </label>
          <input
            id="confirmPassword"
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            placeholder="••••••••"
            required
            disabled={isLoading}
          />
        </div>

        <button type="submit" className="btn-primary btn-block" disabled={isLoading}>
          {isLoading ? (
            <>
              <Loader2 size={18} className="spinner" />
              Creating Account...
            </>
          ) : (
            'Create Account'
          )}
        </button>

        <p className="form-footer">
          Already have an account?{' '}
          <button type="button" onClick={onSwitchToLogin} className="link-button">
            Login
          </button>
        </p>
      </form>
    </div>
  );
};
