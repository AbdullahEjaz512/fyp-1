import type { AnalysisResult } from '../../types';
import { Brain, Activity, AlertTriangle, Info, Shield, TrendingUp } from 'lucide-react';
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

      {/* Ensemble Uncertainty Card */}
      {analysis.ensemble && analysis.ensemble.enabled && (
        <div className="ensemble-card" style={{
          background: 'linear-gradient(135deg, rgba(0, 255, 255, 0.05) 0%, rgba(0, 100, 255, 0.05) 100%)',
          border: '1px solid rgba(0, 255, 255, 0.3)',
          borderRadius: '12px',
          padding: '20px',
          marginBottom: '20px'
        }}>
          <div className="card-header" style={{ marginBottom: '15px' }}>
            <Shield size={24} style={{ color: '#00ffff' }} />
            <h3 style={{ color: '#00ffff' }}>Ensemble AI - Uncertainty Analysis</h3>
            <TrendingUp size={20} style={{ color: '#4ade80', marginLeft: 'auto' }} />
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '15px' }}>
            {/* Segmentation Uncertainty */}
            {analysis.ensemble.segmentation_uncertainty && (
              <div style={{
                background: 'rgba(0, 0, 0, 0.3)',
                borderRadius: '8px',
                padding: '15px',
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}>
                <h4 style={{ fontSize: '0.9em', color: '#00ffff', marginBottom: '10px', fontWeight: '600' }}>
                  Segmentation Quality
                </h4>
                {analysis.ensemble.segmentation_uncertainty.mean_confidence !== undefined && (
                  <div style={{ marginBottom: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ fontSize: '0.85em', color: '#aaa' }}>Confidence</span>
                      <span style={{ fontSize: '0.85em', fontWeight: 'bold' }}>
                        {(analysis.ensemble.segmentation_uncertainty.mean_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{
                        width: `${analysis.ensemble.segmentation_uncertainty.mean_confidence * 100}%`,
                        height: '100%',
                        background: analysis.ensemble.segmentation_uncertainty.mean_confidence > 0.8 ? '#4ade80' : analysis.ensemble.segmentation_uncertainty.mean_confidence > 0.6 ? '#fbbf24' : '#f87171',
                        transition: 'width 0.3s ease'
                      }}></div>
                    </div>
                  </div>
                )}
                {analysis.ensemble.segmentation_uncertainty.mean_entropy !== undefined && (
                  <div style={{ marginBottom: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ fontSize: '0.85em', color: '#aaa' }}>Uncertainty</span>
                      <span style={{ fontSize: '0.85em', fontWeight: 'bold' }}>
                        {(analysis.ensemble.segmentation_uncertainty.mean_entropy * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{
                        width: `${analysis.ensemble.segmentation_uncertainty.mean_entropy * 100}%`,
                        height: '100%',
                        background: analysis.ensemble.segmentation_uncertainty.mean_entropy < 0.2 ? '#4ade80' : analysis.ensemble.segmentation_uncertainty.mean_entropy < 0.4 ? '#fbbf24' : '#f87171',
                        transition: 'width 0.3s ease'
                      }}></div>
                    </div>
                  </div>
                )}
                {analysis.ensemble.segmentation_uncertainty.quality_flags && (
                  <div style={{ marginTop: '12px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                    {analysis.ensemble.segmentation_uncertainty.quality_flags.high_confidence && (
                      <span style={{ fontSize: '0.75em', padding: '3px 8px', borderRadius: '4px', background: 'rgba(74, 222, 128, 0.2)', color: '#4ade80', border: '1px solid rgba(74, 222, 128, 0.3)' }}>
                        ✓ High Confidence
                      </span>
                    )}
                    {analysis.ensemble.segmentation_uncertainty.quality_flags.low_uncertainty && (
                      <span style={{ fontSize: '0.75em', padding: '3px 8px', borderRadius: '4px', background: 'rgba(74, 222, 128, 0.2)', color: '#4ade80', border: '1px solid rgba(74, 222, 128, 0.3)' }}>
                        ✓ Low Uncertainty
                      </span>
                    )}
                    {analysis.ensemble.segmentation_uncertainty.quality_flags.recommended_for_clinical_use && (
                      <span style={{ fontSize: '0.75em', padding: '3px 8px', borderRadius: '4px', background: 'rgba(0, 255, 255, 0.2)', color: '#00ffff', border: '1px solid rgba(0, 255, 255, 0.3)' }}>
                        ✓ Clinical Ready
                      </span>
                    )}
                    {analysis.ensemble.segmentation_uncertainty.quality_flags.requires_expert_review && (
                      <span style={{ fontSize: '0.75em', padding: '3px 8px', borderRadius: '4px', background: 'rgba(251, 191, 36, 0.2)', color: '#fbbf24', border: '1px solid rgba(251, 191, 36, 0.3)' }}>
                        ⚠ Expert Review Needed
                      </span>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Classification Uncertainty */}
            {analysis.ensemble.classification_uncertainty && (
              <div style={{
                background: 'rgba(0, 0, 0, 0.3)',
                borderRadius: '8px',
                padding: '15px',
                border: '1px solid rgba(255, 255, 255, 0.1)'
              }}>
                <h4 style={{ fontSize: '0.9em', color: '#00ffff', marginBottom: '10px', fontWeight: '600' }}>
                  Classification Quality
                </h4>
                {analysis.ensemble.classification_uncertainty.epistemic_uncertainty !== undefined && (
                  <div style={{ marginBottom: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                      <span style={{ fontSize: '0.85em', color: '#aaa' }}>Model Uncertainty</span>
                      <span style={{ fontSize: '0.85em', fontWeight: 'bold' }}>
                        {(analysis.ensemble.classification_uncertainty.epistemic_uncertainty * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                      <div style={{
                        width: `${analysis.ensemble.classification_uncertainty.epistemic_uncertainty * 100}%`,
                        height: '100%',
                        background: analysis.ensemble.classification_uncertainty.epistemic_uncertainty < 0.15 ? '#4ade80' : analysis.ensemble.classification_uncertainty.epistemic_uncertainty < 0.3 ? '#fbbf24' : '#f87171',
                        transition: 'width 0.3s ease'
                      }}></div>
                    </div>
                  </div>
                )}
                {analysis.ensemble.classification_uncertainty.quality_flags && (
                  <div style={{ marginTop: '12px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                    {analysis.ensemble.classification_uncertainty.quality_flags.high_confidence && (
                      <span style={{ fontSize: '0.75em', padding: '3px 8px', borderRadius: '4px', background: 'rgba(74, 222, 128, 0.2)', color: '#4ade80', border: '1px solid rgba(74, 222, 128, 0.3)' }}>
                        ✓ High Confidence
                      </span>
                    )}
                    {analysis.ensemble.classification_uncertainty.quality_flags.low_uncertainty && (
                      <span style={{ fontSize: '0.75em', padding: '3px 8px', borderRadius: '4px', background: 'rgba(74, 222, 128, 0.2)', color: '#4ade80', border: '1px solid rgba(74, 222, 128, 0.3)' }}>
                        ✓ Low Uncertainty
                      </span>
                    )}
                    {analysis.ensemble.classification_uncertainty.quality_flags.recommended_for_clinical_use && (
                      <span style={{ fontSize: '0.75em', padding: '3px 8px', borderRadius: '4px', background: 'rgba(0, 255, 255, 0.2)', color: '#00ffff', border: '1px solid rgba(0, 255, 255, 0.3)' }}>
                        ✓ Clinical Ready
                      </span>
                    )}
                    {analysis.ensemble.classification_uncertainty.quality_flags.requires_expert_review && (
                      <span style={{ fontSize: '0.75em', padding: '3px 8px', borderRadius: '4px', background: 'rgba(251, 191, 36, 0.2)', color: '#fbbf24', border: '1px solid rgba(251, 191, 36, 0.3)' }}>
                        ⚠ Expert Review Needed
                      </span>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          <div style={{ marginTop: '15px', padding: '12px', background: 'rgba(0, 255, 255, 0.05)', borderRadius: '6px', border: '1px solid rgba(0, 255, 255, 0.2)' }}>
            <p style={{ fontSize: '0.85em', color: '#aaa', margin: 0, lineHeight: '1.5' }}>
              <strong style={{ color: '#00ffff' }}>Ensemble AI Technology:</strong> This analysis uses advanced ensemble methods with Test-Time Augmentation and Monte Carlo Dropout to provide uncertainty quantification. Expected improvements: +3-5% segmentation accuracy, +2-4% classification accuracy.
            </p>
          </div>
        </div>
      )}

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
