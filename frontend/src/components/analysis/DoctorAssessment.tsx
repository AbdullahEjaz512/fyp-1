import type { DoctorAssessment } from '../../types';
import { User, Stethoscope, Calendar, FileText } from 'lucide-react';
import './DoctorAssessment.css';

interface DoctorAssessmentProps {
  doctorInfo: {
    doctor_id: number;
    doctor_name: string;
    specialization: string;
    assessment_date: string;
  };
  assessment?: DoctorAssessment;
}

export const DoctorAssessmentView = ({ doctorInfo, assessment }: DoctorAssessmentProps) => {
  if (!assessment) return null;

  return (
    <div className="doctor-assessment">
      <div className="assessment-header">
        <Stethoscope size={24} className="header-icon" />
        <h3>Clinical Assessment</h3>
      </div>

      {/* Doctor Info */}
      <div className="doctor-info-section">
        <div className="doctor-info-item">
          <User size={18} />
          <div>
            <span className="info-label">Assessed by</span>
            <span className="info-value">{doctorInfo.doctor_name}</span>
          </div>
        </div>
        <div className="doctor-info-item">
          <Stethoscope size={18} />
          <div>
            <span className="info-label">Specialization</span>
            <span className="info-value">{doctorInfo.specialization}</span>
          </div>
        </div>
        <div className="doctor-info-item">
          <Calendar size={18} />
          <div>
            <span className="info-label">Assessment Date</span>
            <span className="info-value">
              {new Date(doctorInfo.assessment_date).toLocaleString()}
            </span>
          </div>
        </div>
      </div>

      {/* Clinical Diagnosis */}
      {assessment.diagnosis && (
        <div className="assessment-section diagnosis-section">
          <div className="section-title">
            <FileText size={20} />
            <h4>Clinical Diagnosis</h4>
          </div>
          <p className="diagnosis-text">{assessment.diagnosis}</p>
        </div>
      )}

      {/* Interpretation */}
      {assessment.interpretation && (
        <div className="assessment-section">
          <div className="section-title">
            <h4>Clinical Interpretation</h4>
          </div>
          <p className="assessment-text">{assessment.interpretation}</p>
        </div>
      )}

      {/* Prescription */}
      {assessment.prescription && (
        <div className="assessment-section">
          <div className="section-title">
            <h4>Prescription</h4>
          </div>
          <p className="assessment-text prescription-text">{assessment.prescription}</p>
        </div>
      )}

      {/* Treatment Plan */}
      {assessment.treatment_plan && (
        <div className="assessment-section">
          <div className="section-title">
            <h4>Treatment Plan</h4>
          </div>
          <p className="assessment-text">{assessment.treatment_plan}</p>
        </div>
      )}

      {/* Follow-up Notes */}
      {assessment.follow_up_notes && (
        <div className="assessment-section">
          <div className="section-title">
            <h4>Follow-up Notes</h4>
          </div>
          <p className="assessment-text">{assessment.follow_up_notes}</p>
        </div>
      )}

      {/* Next Appointment */}
      {assessment.next_appointment && (
        <div className="next-appointment">
          <Calendar size={20} />
          <div>
            <span className="appointment-label">Next Appointment</span>
            <span className="appointment-date">
              {new Date(assessment.next_appointment).toLocaleDateString('en-US', {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric',
              })}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
