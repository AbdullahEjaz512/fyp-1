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
  UserPlus,
  Download,
  Lightbulb,
  Eye,
  Box,
  TrendingUp
} from 'lucide-react';
import './ResultsPage.css';

export default function ResultsPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const user = useAuthStore((state) => state.user);
  // Prefer explicit query param, fall back to last viewed file so viz pages work after refresh
  const fileId = searchParams.get('fileId') || localStorage.getItem('lastFileId');
  const [selectedAnalysisIndex, setSelectedAnalysisIndex] = useState(0);
  const [showShareModal, setShowShareModal] = useState(false);
  const [showXAI, setShowXAI] = useState(false);
  const [xaiLoading, setXaiLoading] = useState(false);
  const [xaiData, setXaiData] = useState<any>(null);
  
  const isDoctorRole = user?.role && ['doctor', 'radiologist', 'oncologist'].includes(user.role);

  const { data: resultsData, isLoading, error } = useQuery({
    queryKey: ['analysis-results', fileId],
    queryFn: () => fileService.getAnalysisResults(Number(fileId)),
    enabled: !!fileId,
    retry: 2,
    retryDelay: 1000,
  });

  const generateReport = async () => {
    if (!resultsData || !selectedAnalysis) return;
    
    try {
      const reportData = {
        patient_id: resultsData.patient_id,
        doctor_name: user?.full_name || 'Doctor',
        summary: `Analysis of ${resultsData.filename}`,
        classification: selectedAnalysis.classification.prediction,
        segmentation: {
          volume: `${selectedAnalysis.segmentation.total_volume.mm3.toFixed(2)} mm³`,
          dice: selectedAnalysis.segmentation.metrics?.dice_score?.toFixed(3) || 'N/A'
        },
        notes: selectedAnalysis.doctor_assessment?.interpretation || 'No additional notes'
      };
      
      // Call assistant API to generate PDF report
      const response = await fetch('http://localhost:8000/api/v1/assistant/report/pdf', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        },
        body: JSON.stringify(reportData)
      });
      
      const data = await response.json();
      
      // Download PDF
      const pdfBlob = await fetch(`data:application/pdf;base64,${data.pdf_base64}`).then(r => r.blob());
      const url = window.URL.createObjectURL(pdfBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `report_${resultsData.patient_id}_${Date.now()}.pdf`;
      a.click();
      window.URL.revokeObjectURL(url);
      
    } catch (err) {
      console.error('Report generation failed:', err);
      alert('Failed to generate report. Please try again.');
    }
  };
  
  const loadXAI = async () => {
    if (!fileId || !selectedAnalysis) return;
    
    try {
      setXaiLoading(true);
      
      // Load classification explanation
      const classResponse = await fetch('http://localhost:8000/api/v1/advanced/explain/classification', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        },
        body: JSON.stringify({
          file_id: parseInt(fileId),
          method: 'gradcam'
        })
      });
      
      const classData = await classResponse.json();
      setXaiData(classData);
      setShowXAI(true);
      
    } catch (err) {
      console.error('XAI loading failed:', err);
      alert('Failed to load explainability visualization');
    } finally {
      setXaiLoading(false);
    }
  };
  
  // Reset selected analysis when data changes
  useEffect(() => {
    setSelectedAnalysisIndex(0);
    if (resultsData?.file_id) {
      localStorage.setItem('lastFileId', resultsData.file_id.toString());
    }
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

        {/* Action Buttons */}
        <div className="action-buttons">
          <button className="btn btn-primary" onClick={generateReport}>
            <Download size={18} />
            Generate Report (PDF)
          </button>
          
          <button 
            className="btn btn-secondary" 
            onClick={loadXAI}
            disabled={xaiLoading}
          >
            {xaiLoading ? (
              <>
                <Loader2 size={18} className="spinner" />
                Loading...
              </>
            ) : (
              <>
                <Lightbulb size={18} />
                Explain AI Decision
              </>
            )}
          </button>
        </div>

        {/* Share Case Modal */}
        {showShareModal && (
          <ShareCaseModal
            fileId={Number(fileId)}
            onClose={() => setShowShareModal(false)}
          />
        )}
        
        {/* XAI Visualization Modal */}
        {showXAI && xaiData && (
          <div className="xai-modal" onClick={() => setShowXAI(false)}>
            <div className="xai-modal-content" onClick={(e) => e.stopPropagation()}>
              <div className="xai-header">
                <h3>
                  <Lightbulb size={24} />
                  AI Decision Explanation
                </h3>
                <button onClick={() => setShowXAI(false)} className="close-btn">×</button>
              </div>
              <div className="xai-body">
                <p className="xai-description">
                  This visualization highlights the regions of the MRI scan that most influenced 
                  the AI's classification decision. Warmer colors (red/yellow) indicate higher importance.
                </p>
                {xaiData.heatmap_base64 && (
                  <img 
                    src={`data:image/png;base64,${xaiData.heatmap_base64}`}
                    alt="AI Explanation Heatmap"
                    style={{ width: '100%', borderRadius: '8px' }}
                  />
                )}
                <div className="xai-info">
                  <div className="info-item">
                    <strong>Method:</strong> {xaiData.method || 'Grad-CAM'}
                  </div>
                  <div className="info-item">
                    <strong>Target Class:</strong> {xaiData.target_class || selectedAnalysis.classification.prediction.tumor_type}
                  </div>
                  <div className="info-item">
                    <strong>Confidence:</strong> {selectedAnalysis.classification.prediction.confidence.toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          </div>
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
