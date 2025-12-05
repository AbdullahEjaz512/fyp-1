import type { AnalysisResult } from '../../types';
import { Brain, Activity, AlertTriangle, Info } from 'lucide-react';
import './AnalysisResults.css';

interface AnalysisResultsProps {
  analysis: AnalysisResult;
}

export const AnalysisResults = ({ analysis }: AnalysisResultsProps) => {
  if (!analysis) return null;

  const { segmentation, classification, summary } = analysis;

  // Safety check for required data
  if (!segmentation || !classification || !summary) {
    return (
      <div className="analysis-results">
        <div className="note-card error">
          <AlertTriangle size={20} />
          <p>Analysis data is incomplete or malformed.</p>
        </div>
      </div>
    );
  }

  // Safe access to nested properties
  const ncr = segmentation.regions?.NCR || { volume_mm3: 0, volume_voxels: 0 };
  const ed = segmentation.regions?.ED || { volume_mm3: 0, volume_voxels: 0 };
  const et = segmentation.regions?.ET || { volume_mm3: 0, volume_voxels: 0 };
  
  const totalVolume = segmentation.total_volume || { mm3: 0, voxels: 0 };
  const metrics = segmentation.metrics || { dice_score: 0 };
  
  const prediction = classification.prediction || {
    tumor_type: 'Unknown',
    confidence: 0,
    who_grade: 'Unknown',
    malignancy: 'Unknown'
  };

  const getMalignancyColor = (malignancy: string) => {
    const colors: Record<string, string> = {
      High: 'var(--danger)',
      Medium: 'var(--warning)',
      Low: 'var(--success)',
    };
    return colors[malignancy] || 'var(--text-secondary)';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'var(--success)';
    if (confidence >= 75) return 'var(--primary)';
    if (confidence >= 60) return 'var(--warning)';
    return 'var(--danger)';
  };

  return (
    <div className="analysis-results">
      {/* Summary Card */}
      <div className="summary-card">
        <div className="summary-header">
          <Brain size={32} className="summary-icon" />
          <div>
            <h3 className="summary-title">AI Analysis Results</h3>
            <p className="summary-date">
              {new Date(analysis.timestamp).toLocaleString()}
            </p>
          </div>
        </div>

        <div className="summary-diagnosis">
          <div className="diagnosis-main">
            <span className="diagnosis-label">Diagnosis</span>
            <h2 className="diagnosis-value">{summary.diagnosis}</h2>
          </div>
          <div className="confidence-badge" style={{ background: getConfidenceColor(summary.confidence) }}>
            {summary.confidence.toFixed(1)}% confidence
          </div>
        </div>

        <div className="summary-details-grid">
          <div className="summary-detail">
            <span className="detail-label">WHO Grade</span>
            <span className="detail-value">{summary.who_grade}</span>
          </div>
          <div className="summary-detail">
            <span className="detail-label">Malignancy</span>
            <span 
              className="detail-value" 
              style={{ color: getMalignancyColor(summary.malignancy) }}
            >
              {summary.malignancy}
            </span>
          </div>
          <div className="summary-detail">
            <span className="detail-label">Tumor Volume</span>
            <span className="detail-value">
              {summary.tumor_volume_mm3.toFixed(2)} mm³
            </span>
          </div>
        </div>
      </div>

      {/* Segmentation Results */}
      <div className="segmentation-card">
        <div className="card-header">
          <Activity size={24} />
          <h3>Tumor Segmentation</h3>
        </div>

        <div className="regions-grid">
          <div className="region-card ncr">
            <div className="region-header">
              <div className="region-color"></div>
              <span className="region-name">Necrotic Core (NCR)</span>
            </div>
            <div className="region-stats">
              <div className="region-stat">
                <span className="stat-label">Volume</span>
                <span className="stat-value">
                  {ncr.volume_mm3.toFixed(2)} mm³
                </span>
              </div>
              <div className="region-stat">
                <span className="stat-label">Voxels</span>
                <span className="stat-value">
                  {(ncr.volume_voxels || (ncr as any).voxel_count || 0).toLocaleString()}
                </span>
              </div>
            </div>
          </div>

          <div className="region-card ed">
            <div className="region-header">
              <div className="region-color"></div>
              <span className="region-name">Edema (ED)</span>
            </div>
            <div className="region-stats">
              <div className="region-stat">
                <span className="stat-label">Volume</span>
                <span className="stat-value">
                  {ed.volume_mm3.toFixed(2)} mm³
                </span>
              </div>
              <div className="region-stat">
                <span className="stat-label">Voxels</span>
                <span className="stat-value">
                  {(ed.volume_voxels || (ed as any).voxel_count || 0).toLocaleString()}
                </span>
              </div>
            </div>
          </div>

          <div className="region-card et">
            <div className="region-header">
              <div className="region-color"></div>
              <span className="region-name">Enhancing Tumor (ET)</span>
            </div>
            <div className="region-stats">
              <div className="region-stat">
                <span className="stat-label">Volume</span>
                <span className="stat-value">
                  {et.volume_mm3.toFixed(2)} mm³
                </span>
              </div>
              <div className="region-stat">
                <span className="stat-label">Voxels</span>
                <span className="stat-value">
                  {(et.volume_voxels || (et as any).voxel_count || 0).toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="total-volume-card">
          <div className="total-label">Total Tumor Volume</div>
          <div className="total-value">
            {totalVolume.mm3.toFixed(2)} mm³
          </div>
          <div className="total-voxels">
            {totalVolume.voxels.toLocaleString()} voxels
          </div>
        </div>

        <div className="metrics-row">
          <div className="metric-item">
            <span className="metric-label">Dice Score</span>
            <span className="metric-value">{metrics.dice_score.toFixed(3)}</span>
          </div>
          {metrics.hausdorff_distance !== undefined && (
            <div className="metric-item">
              <span className="metric-label">Hausdorff Distance</span>
              <span className="metric-value">{metrics.hausdorff_distance.toFixed(2)} mm</span>
            </div>
          )}
        </div>
      </div>

      {/* Classification Results */}
      <div className="classification-card">
        <div className="card-header">
          <Brain size={24} />
          <h3>Tumor Classification</h3>
        </div>

        <div className="classification-main">
          <div className="class-prediction">
            <span className="class-label">Predicted Type</span>
            <h3 className="class-value">{prediction.tumor_type}</h3>
            <div className="confidence-bar">
              <div 
                className="confidence-fill" 
                style={{ 
                  width: `${prediction.confidence}%`,
                  background: getConfidenceColor(prediction.confidence)
                }}
              ></div>
            </div>
            <span className="confidence-text">
              {prediction.confidence.toFixed(1)}% confidence
            </span>
          </div>

          <div className="classification-details">
            <div className="class-detail-item">
              <AlertTriangle size={18} />
              <div>
                <span className="class-detail-label">Malignancy Level</span>
                <span 
                  className="class-detail-value"
                  style={{ color: getMalignancyColor(prediction.malignancy) }}
                >
                  {prediction.malignancy}
                </span>
              </div>
            </div>
            <div className="class-detail-item">
              <Info size={18} />
              <div>
                <span className="class-detail-label">WHO Grade</span>
                <span className="class-detail-value">{prediction.who_grade}</span>
              </div>
            </div>
          </div>
        </div>

        {classification.class_probabilities && (
          <div className="probabilities-section">
            <h4 className="probabilities-title">All Classifications</h4>
            <div className="probabilities-list">
              {Object.entries(classification.class_probabilities)
                .sort(([, a], [, b]) => b - a)
                .map(([tumorType, probability]) => (
                  <div key={tumorType} className="probability-item">
                    <span className="probability-name">{tumorType}</span>
                    <div className="probability-bar-container">
                      <div 
                        className="probability-bar" 
                        style={{ width: `${(probability * 100).toFixed(1)}%` }}
                      ></div>
                    </div>
                    <span className="probability-value">
                      {(probability * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>

      {/* Note */}
      {analysis.note && (
        <div className="note-card">
          <Info size={20} />
          <p>{analysis.note}</p>
        </div>
      )}
    </div>
  );
};
