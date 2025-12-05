import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { collaborationService } from '../../services/file.service';
import { Users, UserMinus, CheckCircle, Star, Stethoscope } from 'lucide-react';
import './Collaboration.css';

interface CollaborationPanelProps {
  fileId: number;
}

export const CollaborationPanel = ({ fileId }: CollaborationPanelProps) => {
  const queryClient = useQueryClient();

  const { data, isLoading, error } = useQuery({
    queryKey: ['collaborators', fileId],
    queryFn: () => collaborationService.getCollaborators(fileId),
  });

  const removeMutation = useMutation({
    mutationFn: (doctorId: number) => collaborationService.removeCollaborator(fileId, doctorId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['collaborators', fileId] });
    },
  });

  if (isLoading) {
    return (
      <div className="collaboration-panel">
        <div className="panel-header">
          <h4><Users size={18} /> Collaborating Doctors</h4>
        </div>
        <div className="loading-state">Loading collaborators...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="collaboration-panel">
        <div className="panel-header">
          <h4><Users size={18} /> Collaborating Doctors</h4>
        </div>
        <div className="error-state">Failed to load collaborators</div>
      </div>
    );
  }

  const collaborators = data?.collaborators || [];

  return (
    <div className="collaboration-panel">
      <div className="panel-header">
        <h4><Users size={18} /> Collaborating Doctors ({collaborators.length})</h4>
      </div>

      {collaborators.length === 0 ? (
        <div className="empty-state">
          <Users size={24} />
          <p>No collaborators yet</p>
          <span>Share this case to invite other doctors</span>
        </div>
      ) : (
        <div className="collaborators-list">
          {collaborators.map((doctor) => (
            <div key={doctor.id} className={`collaborator-item ${doctor.is_current_user ? 'current-user' : ''}`}>
              <div className="collaborator-avatar">
                {doctor.name.charAt(0).toUpperCase()}
              </div>
              <div className="collaborator-info">
                <div className="collaborator-name">
                  {doctor.name}
                  {doctor.is_primary && (
                    <span className="badge primary">
                      <Star size={12} /> Primary
                    </span>
                  )}
                  {doctor.is_current_user && (
                    <span className="badge current">You</span>
                  )}
                </div>
                <div className="collaborator-details">
                  {doctor.specialization && (
                    <span className="specialization">
                      <Stethoscope size={12} /> {doctor.specialization}
                    </span>
                  )}
                  {doctor.has_analyzed && (
                    <span className="analyzed">
                      <CheckCircle size={12} /> Analyzed
                    </span>
                  )}
                </div>
              </div>
              {!doctor.is_primary && !doctor.is_current_user && (
                <button
                  className="remove-btn"
                  onClick={() => removeMutation.mutate(doctor.id)}
                  disabled={removeMutation.isPending}
                  title="Remove collaborator"
                >
                  <UserMinus size={16} />
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
