import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { fileService } from '../services/file.service';
import { AnalysisResults } from '../components/analysis/AnalysisResults';
import { DoctorAssessmentView } from '../components/analysis/DoctorAssessment';
import { DoctorAssessmentForm } from '../components/analysis/DoctorAssessmentForm';
import { ShareCaseModal, CollaborationPanel, DiscussionThread } from '../components/collaboration';
import { 
  FileText, 
  AlertCircle, 
  Loader2, 
  ChevronLeft,
  User,
  Calendar,
  UserPlus
} from 'lucide-react';
import './ResultsPage.css';

export default function ResultsPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const user = useAuthStore((state) => state.user);
  const fileId = searchParams.get('fileId');
  const [selectedAnalysisIndex, setSelectedAnalysisIndex] = useState(0);
  const [showShareModal, setShowShareModal] = useState(false);
  
  const isDoctorRole = user?.role && ['doctor', 'radiologist', 'oncologist'].includes(user.role);

  const { data: resultsData, isLoading, error } = useQuery({
    queryKey: ['analysis-results', fileId],
    queryFn: () => fileService.getAnalysisResults(Number(fileId)),
    enabled: !!fileId,
    retry: 2,
    retryDelay: 1000,
  });

  // Reset selected analysis when data changes
  useEffect(() => {
    setSelectedAnalysisIndex(0);
  }, [resultsData]);

  if (!fileId) {
    return (
      <div className="results-page">
        <div className="container">
          <div className="empty-state-large">
            <FileText size={64} />
            <h2>No Analysis Selected</h2>
            <p>Please select a file from the upload page to view results</p>
            <button onClick={() => navigate('/upload')} className="btn btn-primary">
              Go to Upload Page
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="results-page">
        <div className="container">
          <div className="loading-state">
            <Loader2 size={64} className="spinner" />
            <p>Loading analysis results...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return (
      <div className="results-page">
        <div className="container">
          <div className="error-state">
            <AlertCircle size={64} />
            <h2>Error Loading Results</h2>
            <p>{errorMessage}</p>
            <p className="error-hint">
              Make sure the file has been analyzed. If the problem persists, try analyzing the file again.
            </p>
            <button onClick={() => navigate('/upload')} className="btn btn-secondary">
              Back to Upload
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!resultsData || !resultsData.analyses || resultsData.analyses.length === 0) {
    return (
      <div className="results-page">
        <div className="container">
          <div className="empty-state-large">
            <AlertCircle size={64} />
            <h2>No Analysis Results Found</h2>
            <p>This file hasn't been analyzed yet or the analysis is still in progress.</p>
            <div className="empty-actions">
              <button onClick={() => navigate('/upload')} className="btn btn-primary">
                Go to Upload Page
              </button>
              <p className="empty-hint">
                Tip: Make sure to click the "Analyze" button after uploading your file.
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const selectedAnalysis = resultsData.analyses[selectedAnalysisIndex];

  if (!selectedAnalysis) {
    return (
      <div className="results-page">
        <div className="container">
          <div className="error-state">
            <AlertCircle size={64} />
            <h2>Analysis Not Found</h2>
            <p>The selected analysis could not be found.</p>
            <button onClick={() => navigate('/upload')} className="btn btn-secondary">
              Back to Upload
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="results-page">
      <div className="container">
        {/* Header */}
        <div className="results-header">
          <button onClick={() => navigate('/upload')} className="btn-back">
            <ChevronLeft size={20} />
            Back to Files
          </button>
          
          <div className="file-info-header">
            <FileText size={24} className="file-icon" />
            <div>
              <h1 className="page-title">{resultsData.filename}</h1>
              <div className="file-meta">
                <span>
                  <User size={14} />
                  Patient: {resultsData.patient_id}
                </span>
                <span>
                  <Calendar size={14} />
                  {resultsData.total_analyses} analysis result{resultsData.total_analyses !== 1 ? 's' : ''}
                </span>
              </div>
            </div>
          </div>

          {/* Share Case Button */}
          {isDoctorRole && (
            <button 
              className="share-case-btn"
              onClick={() => setShowShareModal(true)}
            >
              <UserPlus size={18} />
              Share Case
            </button>
          )}
        </div>

        {/* Share Case Modal */}
        {showShareModal && (
          <ShareCaseModal
            fileId={Number(fileId)}
            onClose={() => setShowShareModal(false)}
          />
        )}

        {/* Multiple Analysis Selector */}
        {resultsData.analyses.length > 1 && (
          <div className="analysis-selector">
            <h3>Multiple Doctor Assessments Available</h3>
            <div className="analysis-tabs">
              {resultsData.analyses.map((analysis, index) => (
                <button
                  key={analysis.analysis_id}
                  onClick={() => setSelectedAnalysisIndex(index)}
                  className={`analysis-tab ${index === selectedAnalysisIndex ? 'active' : ''}`}
                >
                  <div className="tab-doctor">
                    Dr. {analysis.doctor_info?.doctor_name || 'AI Analysis'}
                  </div>
                  <div className="tab-spec">
                    {analysis.doctor_info?.specialization || 'Automated'}
                  </div>
                  <div className="tab-date">
                    {new Date(analysis.timestamp).toLocaleDateString()}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Analysis Results */}
        <AnalysisResults analysis={selectedAnalysis} />

        {/* Doctor Assessment View */}
        {selectedAnalysis.doctor_info && selectedAnalysis.doctor_assessment && (
          <DoctorAssessmentView
            doctorInfo={selectedAnalysis.doctor_info}
            assessment={selectedAnalysis.doctor_assessment}
          />
        )}

        {/* Doctor Assessment Form (for doctors to add/edit assessment) */}
        {isDoctorRole && (
          <DoctorAssessmentForm
            fileId={Number(fileId)}
            existingAssessment={selectedAnalysis.doctor_assessment}
          />
        )}

        {/* Collaboration Section */}
        {isDoctorRole && (
          <div className="collaboration-section">
            <h2 className="section-title">
              <UserPlus size={20} />
              Collaboration
            </h2>
            <div className="collaboration-grid">
              <CollaborationPanel fileId={Number(fileId)} />
              <DiscussionThread fileId={Number(fileId)} />
            </div>
          </div>
        )}

        {/* Footer Info */}
        {selectedAnalysis.doctor_info && (
          <div className="results-footer">
            <p>
              Analysis performed on {new Date(selectedAnalysis.timestamp).toLocaleDateString()} 
              by Dr. {selectedAnalysis.doctor_info.doctor_name} 
              ({selectedAnalysis.doctor_info.specialization})
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
