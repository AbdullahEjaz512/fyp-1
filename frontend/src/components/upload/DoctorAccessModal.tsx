import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { doctorService } from '../../services/file.service';
import { X, Users, Stethoscope, Check, Loader2 } from 'lucide-react';
import './DoctorAccessModal.css';

interface DoctorAccessModalProps {
  fileId: number;
  filename: string;
  onClose: () => void;
}

export const DoctorAccessModal = ({ fileId, filename, onClose }: DoctorAccessModalProps) => {
  const queryClient = useQueryClient();
  const [selectedDoctors, setSelectedDoctors] = useState<number[]>([]);
  const [searchTerm, setSearchTerm] = useState('');

  const { data: doctors, isLoading: loadingDoctors } = useQuery({
    queryKey: ['doctors'],
    queryFn: doctorService.listDoctors,
  });

  const { data: currentAccess } = useQuery({
    queryKey: ['file-access', fileId],
    queryFn: () => doctorService.getFileAccess(fileId),
  });

  useEffect(() => {
    if (currentAccess?.doctors_with_access) {
      setSelectedDoctors(currentAccess.doctors_with_access.map(d => d.doctor_id));
    }
  }, [currentAccess]);

  const grantAccessMutation = useMutation({
    mutationFn: (doctorIds: number[]) => doctorService.grantAccess(fileId, doctorIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['file-access', fileId] });
      queryClient.invalidateQueries({ queryKey: ['files'] });
      alert('Access granted successfully!');
      onClose();
    },
    onError: (error: any) => {
      alert(error.response?.data?.detail || 'Failed to grant access');
    },
  });

  const handleToggleDoctor = (doctorId: number) => {
    setSelectedDoctors(prev => 
      prev.includes(doctorId)
        ? prev.filter(id => id !== doctorId)
        : [...prev, doctorId]
    );
  };

  const handleSave = () => {
    grantAccessMutation.mutate(selectedDoctors);
  };

  const filteredDoctors = doctors?.filter(doctor => 
    doctor.full_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    doctor.specialization.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title-group">
            <Users size={24} className="modal-icon" />
            <div>
              <h2>Grant Doctor Access</h2>
              <p className="modal-subtitle">{filename}</p>
            </div>
          </div>
          <button onClick={onClose} className="btn-close">
            <X size={24} />
          </button>
        </div>

        <div className="modal-body">
          <div className="search-box">
            <input
              type="text"
              placeholder="Search doctors by name or specialization..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>

          {loadingDoctors ? (
            <div className="loading-state">
              <Loader2 size={32} className="spinner" />
              <p>Loading doctors...</p>
            </div>
          ) : filteredDoctors.length === 0 ? (
            <div className="empty-state">
              <Users size={48} />
              <p>No doctors found</p>
            </div>
          ) : (
            <div className="doctors-list">
              {filteredDoctors.map((doctor) => (
                <div
                  key={doctor.user_id}
                  className={`doctor-item ${selectedDoctors.includes(doctor.user_id) ? 'selected' : ''}`}
                  onClick={() => handleToggleDoctor(doctor.user_id)}
                >
                  <div className="doctor-info">
                    <div className="doctor-avatar">
                      <Stethoscope size={20} />
                    </div>
                    <div className="doctor-details">
                      <h4 className="doctor-name">{doctor.full_name}</h4>
                      <p className="doctor-spec">{doctor.specialization}</p>
                      {doctor.institution && (
                        <p className="doctor-license">{doctor.institution}</p>
                      )}
                    </div>
                  </div>
                  <div className="doctor-checkbox">
                    {selectedDoctors.includes(doctor.user_id) && (
                      <Check size={20} />
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="modal-footer">
          <div className="selected-count">
            {selectedDoctors.length} doctor{selectedDoctors.length !== 1 ? 's' : ''} selected
          </div>
          <div className="modal-actions">
            <button onClick={onClose} className="btn btn-secondary">
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={grantAccessMutation.isPending || selectedDoctors.length === 0}
              className="btn btn-primary"
            >
              {grantAccessMutation.isPending ? (
                <>
                  <Loader2 size={20} className="spinner" />
                  Saving...
                </>
              ) : (
                <>
                  <Check size={20} />
                  Grant Access
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
