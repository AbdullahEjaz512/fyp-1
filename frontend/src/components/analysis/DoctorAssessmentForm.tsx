import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { fileService } from '../../services/file.service';
import { Stethoscope, Save, Loader2, FileText, Pill, Calendar } from 'lucide-react';
import './DoctorAssessmentForm.css';

interface DoctorAssessmentFormProps {
  fileId: number;
  existingAssessment?: {
    doctor_interpretation?: string;
    clinical_diagnosis?: string;
    prescription?: string;
    treatment_plan?: string;
    follow_up_notes?: string;
    next_appointment?: string;
  };
}

export const DoctorAssessmentForm = ({ fileId, existingAssessment }: DoctorAssessmentFormProps) => {
  const queryClient = useQueryClient();
  const [formData, setFormData] = useState({
    doctor_interpretation: existingAssessment?.doctor_interpretation || '',
    clinical_diagnosis: existingAssessment?.clinical_diagnosis || '',
    prescription: existingAssessment?.prescription || '',
    treatment_plan: existingAssessment?.treatment_plan || '',
    follow_up_notes: existingAssessment?.follow_up_notes || '',
    next_appointment: existingAssessment?.next_appointment || '',
  });

  const saveMutation = useMutation({
    mutationFn: () => fileService.addDoctorAssessment(fileId, formData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['analysis-results', fileId] });
      alert('Assessment saved successfully!');
    },
    onError: (error: any) => {
      alert(error.response?.data?.detail || 'Failed to save assessment');
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    saveMutation.mutate();
  };

  const handleChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <form className="doctor-assessment-form" onSubmit={handleSubmit}>
      <div className="form-header">
        <Stethoscope size={24} className="header-icon" />
        <h3>Add Clinical Assessment</h3>
      </div>

      <div className="form-content">
        {/* Clinical Interpretation */}
        <div className="form-group">
          <label htmlFor="interpretation">
            <FileText size={18} />
            Clinical Interpretation
          </label>
          <textarea
            id="interpretation"
            value={formData.doctor_interpretation}
            onChange={(e) => handleChange('doctor_interpretation', e.target.value)}
            placeholder="Your professional interpretation of the AI analysis results..."
            rows={4}
          />
        </div>

        {/* Clinical Diagnosis */}
        <div className="form-group">
          <label htmlFor="diagnosis">
            <Stethoscope size={18} />
            Clinical Diagnosis <span className="required">*</span>
          </label>
          <input
            type="text"
            id="diagnosis"
            value={formData.clinical_diagnosis}
            onChange={(e) => handleChange('clinical_diagnosis', e.target.value)}
            placeholder="e.g., Glioblastoma Grade IV confirmed"
            required
          />
        </div>

        {/* Prescription */}
        <div className="form-group">
          <label htmlFor="prescription">
            <Pill size={18} />
            Prescription
          </label>
          <textarea
            id="prescription"
            value={formData.prescription}
            onChange={(e) => handleChange('prescription', e.target.value)}
            placeholder="Prescribed medications and dosage..."
            rows={3}
          />
        </div>

        {/* Treatment Plan */}
        <div className="form-group">
          <label htmlFor="treatment">
            <FileText size={18} />
            Treatment Plan
          </label>
          <textarea
            id="treatment"
            value={formData.treatment_plan}
            onChange={(e) => handleChange('treatment_plan', e.target.value)}
            placeholder="Recommended treatment approach and procedures..."
            rows={4}
          />
        </div>

        {/* Follow-up Notes */}
        <div className="form-group">
          <label htmlFor="followup">
            <FileText size={18} />
            Follow-up Notes
          </label>
          <textarea
            id="followup"
            value={formData.follow_up_notes}
            onChange={(e) => handleChange('follow_up_notes', e.target.value)}
            placeholder="Additional notes and follow-up instructions..."
            rows={3}
          />
        </div>

        {/* Next Appointment */}
        <div className="form-group">
          <label htmlFor="appointment">
            <Calendar size={18} />
            Next Appointment
          </label>
          <input
            type="date"
            id="appointment"
            value={formData.next_appointment}
            onChange={(e) => handleChange('next_appointment', e.target.value)}
          />
        </div>
      </div>

      <div className="form-actions">
        <button
          type="submit"
          className="btn btn-primary"
          disabled={saveMutation.isPending || !formData.clinical_diagnosis.trim()}
        >
          {saveMutation.isPending ? (
            <>
              <Loader2 size={20} className="spinner" />
              Saving...
            </>
          ) : (
            <>
              <Save size={20} />
              Save Assessment
            </>
          )}
        </button>
      </div>
    </form>
  );
};
