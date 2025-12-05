import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { collaborationService } from '../../services/file.service';
import { X, UserPlus, Send, CheckCircle, AlertCircle } from 'lucide-react';
import './Collaboration.css';

interface ShareCaseModalProps {
  fileId: number;
  onClose: () => void;
}

export const ShareCaseModal = ({ fileId, onClose }: ShareCaseModalProps) => {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [success, setSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const shareMutation = useMutation({
    mutationFn: () => collaborationService.shareCase(fileId, email, message || undefined),
    onSuccess: (data) => {
      setSuccess(`Case shared with ${data.collaborator.name}`);
      setEmail('');
      setMessage('');
      queryClient.invalidateQueries({ queryKey: ['collaborators', fileId] });
      setTimeout(() => {
        setSuccess(null);
      }, 3000);
    },
    onError: (err: any) => {
      setError(err.response?.data?.detail || 'Failed to share case');
      setTimeout(() => setError(null), 5000);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) {
      setError('Please enter an email address');
      return;
    }
    shareMutation.mutate();
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content share-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3><UserPlus size={20} /> Share Case for Collaboration</h3>
          <button className="close-btn" onClick={onClose}>
            <X size={20} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="share-form">
          {success && (
            <div className="alert alert-success">
              <CheckCircle size={18} />
              {success}
            </div>
          )}

          {error && (
            <div className="alert alert-error">
              <AlertCircle size={18} />
              {error}
            </div>
          )}

          <div className="form-group">
            <label htmlFor="email">Doctor's Email</label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Enter colleague's email"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="message">Message (Optional)</label>
            <textarea
              id="message"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Add a note for the collaborating doctor..."
              rows={3}
            />
          </div>

          <div className="modal-actions">
            <button type="button" className="btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button 
              type="submit" 
              className="btn-primary"
              disabled={shareMutation.isPending}
            >
              {shareMutation.isPending ? (
                'Sharing...'
              ) : (
                <>
                  <Send size={16} /> Share Case
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
