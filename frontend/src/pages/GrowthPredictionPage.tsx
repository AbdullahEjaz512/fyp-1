/**
 * Module 5: Tumor Growth Prediction Page
 * LSTM-based growth trajectory prediction and risk assessment
 */

import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { useAuthStore } from '../store/authStore';
import { advancedService } from '../services/advanced.service';
import { fileService } from '../services/file.service';
import { 
  TrendingUp, 
  AlertTriangle, 
  Calendar, 
  Activity,
  Loader2,
  Info,
  CheckCircle,
  Clock
} from 'lucide-react';
import { HelpTooltip } from '../components/common/HelpTooltip';
import './GrowthPredictionPage.css';
import '../components/common/InfoBanner.css';

export default function GrowthPredictionPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const user = useAuthStore((state) => state.user);
  const patientId = searchParams.get('patientId') || user?.patient_id;
  
  const [loading, setLoading] = useState(false);
  const [predictionData, setPredictionData] = useState<any>(null);
  const [historicalScans, setHistoricalScans] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  
  // Query all files to get unique patient IDs
  const { data: allFiles } = useQuery({
    queryKey: ['files'],
    queryFn: fileService.listFiles,
    enabled: !patientId,
  });

  useEffect(() => {
    if (patientId) {
      loadHistoricalScans();
    }
  }, [patientId]);

  const loadHistoricalScans = async () => {
    try {
      setLoading(true);
      const response = await fileService.listFiles();
      
      // Get all analyzed files for this patient
      const analyzedFiles = response.filter((f: any) => 
        f.status === 'analyzed' && f.patient_id === patientId
      );
      
      setHistoricalScans(analyzedFiles);
      setError(null);
    } catch (err: any) {
      setError('Failed to load historical scans');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const predictGrowth = async () => {
    console.log('=== PREDICT GROWTH CALLED ===');
    console.log('Patient ID:', patientId);
    console.log('Historical scans count:', historicalScans.length);
    
    if (historicalScans.length < 2) {
      setError('Need at least 2 historical scans for growth prediction');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      console.log('Starting growth prediction...');

      // Prepare historical scan data
      const scans = await Promise.all(
        historicalScans.map(async (scan) => {
          const results = await fileService.getAnalysisResults(scan.file_id);
          const analysis = results.analyses[0];
          
          return {
            volume: analysis.segmentation.total_volume.mm3,
            mean_intensity: 120 + Math.random() * 20, // Placeholder
            std_intensity: 25 + Math.random() * 5,
            max_diameter: Math.cbrt(analysis.segmentation.total_volume.mm3 * 0.75 / Math.PI) * 2,
            surface_area: analysis.segmentation.total_volume.mm3 * 0.8,
            sphericity: 0.7 + Math.random() * 0.2,
            compactness: 0.6 + Math.random() * 0.2,
            location_x: 64, location_y: 64, location_z: 64,
            timestamp: scan.upload_date
          };
        })
      );

      console.log('Prepared scans data:', scans);

      const payload = {
        patient_id: patientId,
        historical_scans: scans,
        prediction_steps: 3
      };
      
      console.log('Sending prediction request with payload:', payload);
      const response = await advancedService.predictGrowth(payload);
      console.log('Prediction response:', response);
      console.log('Response data:', response.data);

      if (!response.data) {
        throw new Error('No data received from prediction service');
      }

      setPredictionData(response.data);
      console.log('Prediction data set successfully');
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Growth prediction failed. Please try again.';
      setError(errorMessage);
      console.error('Growth prediction error:', err);
      console.error('Error details:', {
        status: err.response?.status,
        data: err.response?.data,
        message: err.message,
        stack: err.stack
      });
    } finally {
      setLoading(false);
      console.log('=== PREDICT GROWTH FINISHED ===');
    }
  };

  if (!patientId) {
    // Group files by patient_id and count analyzed scans
    const patientGroups = allFiles?.reduce((acc: any, file: any) => {
      const pid = file.patient_id;
      if (!pid) return acc;
      if (!acc[pid]) {
        acc[pid] = { patient_id: pid, files: [], analyzedCount: 0 };
      }
      acc[pid].files.push(file);
      if (file.status === 'analyzed') {
        acc[pid].analyzedCount++;
      }
      return acc;
    }, {}) || {};
    
    const patients = Object.values(patientGroups).filter((p: any) => p.analyzedCount >= 2);
    
    return (
      <div className="growth-page">
        <div className="container">
          <div className="page-header" style={{ textAlign: 'center', marginBottom: '2rem' }}>
            <h1>
              <TrendingUp size={32} />
              Tumor Growth Prediction
              <HelpTooltip 
                title="How Growth Prediction Works" 
                content="Our LSTM (Long Short-Term Memory) AI model analyzes your historical scans to predict how the tumor will grow over time. This helps doctors plan treatment and monitor progression. Requires at least 2 scans taken at different times." 
              />
            </h1>
            <p>AI-powered analysis of tumor progression over time</p>
          </div>
          
          {patients.length === 0 ? (
            <div className="error-state">
              <AlertTriangle size={64} />
              <h2>No Eligible Patients</h2>
              <p>Growth prediction requires at least 2 analyzed scans for a patient.</p>
              <button onClick={() => navigate('/upload')} className="btn btn-primary" style={{ marginTop: '1rem' }}>
                Upload Scans
              </button>
            </div>
          ) : (
            <div style={{ maxWidth: '800px', margin: '0 auto' }}>
              <h3 style={{ marginBottom: '1.5rem' }}>Select a Patient:</h3>
              <div style={{ display: 'grid', gap: '1rem' }}>
                {patients.map((patient: any) => (
                  <div
                    key={patient.patient_id}
                    onClick={() => navigate(`/growth-prediction?patientId=${patient.patient_id}`)}
                    style={{
                      background: 'white',
                      padding: '1.5rem',
                      borderRadius: '8px',
                      cursor: 'pointer',
                      border: '2px solid #e2e8f0',
                      transition: 'all 0.2s',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = '#667eea';
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 4px 12px rgba(102, 126, 234, 0.2)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = '#e2e8f0';
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = 'none';
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', justifyContent: 'space-between' }}>
                      <div>
                        <div style={{ fontWeight: 600, fontSize: '1.1rem', marginBottom: '0.5rem' }}>
                          Patient ID: {patient.patient_id}
                        </div>
                        <div style={{ fontSize: '0.875rem', color: '#64748b' }}>
                          {patient.analyzedCount} analyzed scans available
                        </div>
                      </div>
                      <TrendingUp size={24} color="#667eea" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  const getRiskColor = (risk: string) => {
    switch (risk?.toLowerCase()) {
      case 'high': return '#ff4444';
      case 'medium': return '#ffaa00';
      case 'low': return '#00cc66';
      default: return '#666';
    }
  };

  return (
    <div className="growth-page">
      <div className="container">
        <div className="page-header">
          <h1>
            <TrendingUp size={32} />
            Tumor Growth Prediction
            <HelpTooltip 
              title="How Growth Prediction Works" 
              content="Our LSTM (Long Short-Term Memory) AI model analyzes your historical scans to predict how the tumor will grow over time. This helps doctors plan treatment and monitor progression. Requires at least 2 scans taken at different times." 
            />
          </h1>
          <p>AI-powered analysis of tumor progression over time</p>
        </div>

        {/* Instructional Banner */}
        {!predictionData && historicalScans.length < 2 && (
          <div className="info-banner">
            <Info size={24} />
            <div>
              <h4>How to Use Growth Prediction:</h4>
              <ol>
                <li>Upload at least <strong>2 MRI scans</strong> of the same patient taken at different times</li>
                <li>Make sure both scans have been <strong>analyzed</strong> (check Results page)</li>
                <li>Return here and click <strong>"Predict Growth"</strong></li>
                <li>View the predicted tumor growth trajectory for the next 3-6 months</li>
              </ol>
            </div>
          </div>
        )}

        {/* Patient Info */}
        <div className="patient-info-card">
          <div className="info-header">
            <Activity size={24} />
            <h3>Patient: {patientId}</h3>
          </div>
          <div className="scan-summary">
            <div className="summary-item">
              <Calendar size={18} />
              <span>{historicalScans.length} Historical Scans</span>
            </div>
            {historicalScans.length >= 2 ? (
              <CheckCircle size={18} color="#00cc66" />
            ) : (
              <Clock size={18} color="#ffaa00" />
            )}
            <span>
              {historicalScans.length >= 2 
                ? 'Ready for prediction' 
                : `Need ${2 - historicalScans.length} more scan(s)`}
            </span>
          </div>
        </div>

        {/* Historical Scans List */}
        <div className="scans-card">
          <h3>Historical Scans</h3>
          {historicalScans.length === 0 ? (
            <p className="empty-message">No analyzed scans found for this patient</p>
          ) : (
            <div className="scans-list">
              {historicalScans.map((scan, idx) => (
                <div key={scan.file_id} className="scan-item">
                  <div className="scan-number">#{idx + 1}</div>
                  <div className="scan-details">
                    <div className="scan-name">{scan.filename}</div>
                    <div className="scan-date">
                      {new Date(scan.upload_date).toLocaleDateString()}
                    </div>
                  </div>
                  <button 
                    className="btn-sm"
                    onClick={() => navigate(`/results?fileId=${scan.file_id}`)}
                  >
                    View Results
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Error Display - Show prominently above action */}
        {error && (
          <div className="error-banner" style={{ 
            background: '#fee', 
            border: '2px solid #f44', 
            padding: '1.5rem', 
            borderRadius: '8px',
            marginBottom: '1rem',
            display: 'flex',
            alignItems: 'center',
            gap: '1rem'
          }}>
            <AlertTriangle size={24} color="#f44" />
            <div>
              <strong>Prediction Failed:</strong>
              <p style={{ margin: '0.5rem 0 0 0' }}>{error}</p>
            </div>
          </div>
        )}

        {/* Predict Button */}
        <div className="action-section">
          <button
            className="btn-primary btn-lg"
            onClick={predictGrowth}
            disabled={loading || historicalScans.length < 2}
          >
            {loading ? (
              <>
                <Loader2 size={20} className="spinner" />
                Analyzing...
              </>
            ) : (
              <>
                <TrendingUp size={20} />
                Predict Growth Trajectory
              </>
            )}
          </button>
          {historicalScans.length < 2 && (
            <p className="help-text">
              <Info size={16} />
              At least 2 scans are required to predict tumor growth patterns
            </p>
          )}
        </div>

        {/* Prediction Results */}
        {predictionData && (
          <div className="results-section">
            <h2>Growth Prediction Results</h2>

            {/* Risk Assessment */}
            <div 
              className="risk-card"
              style={{ borderColor: getRiskColor(predictionData.risk_level) }}
            >
              <div className="risk-header">
                <AlertTriangle size={32} color={getRiskColor(predictionData.risk_level)} />
                <div>
                  <h3>Risk Assessment</h3>
                  <div 
                    className="risk-badge"
                    style={{ background: getRiskColor(predictionData.risk_level) }}
                  >
                    {predictionData.risk_level} Risk
                  </div>
                </div>
              </div>
              <div className="risk-details">
                <div className="risk-stat">
                  <span>Growth Rate</span>
                  <strong>
                    {typeof predictionData.growth_rate === 'number'
                      ? `${predictionData.growth_rate.toFixed(2)}%`
                      : 'N/A'}
                  </strong>
                </div>
              </div>
            </div>

            {/* Predictions Table */}
            <div className="predictions-card">
              <h3>Future Volume Predictions</h3>
              <div className="predictions-table">
                <div className="table-header">
                  <div>Time Point</div>
                  <div>Predicted Volume</div>
                  <div>Confidence Interval</div>
                </div>
                {predictionData.predictions?.map((pred: number, idx: number) => {
                  const interval = predictionData.confidence_intervals?.[idx];
                  const predValue = typeof pred === 'number' ? pred.toFixed(2) : 'N/A';
                  const intervalText = interval && typeof interval[0] === 'number' && typeof interval[1] === 'number'
                    ? `${interval[0].toFixed(2)} - ${interval[1].toFixed(2)} mm³`
                    : 'N/A';

                  return (
                    <div key={idx} className="table-row">
                      <div>Step {idx + 1}</div>
                      <div>{predValue} mm³</div>
                      <div>{intervalText}</div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Historical Volumes */}
            <div className="history-card">
              <h3>Historical Volumes</h3>
              <div className="volume-chart">
                {predictionData.historical_volumes?.map((vol: number, idx: number) => {
                  const volumes = predictionData.historical_volumes || [];
                  const numericVolumes = volumes.filter((v: any): v is number => typeof v === 'number');
                  const maxVol = Math.max(...(numericVolumes.length ? numericVolumes : [1]));
                  const safeVol = typeof vol === 'number' ? vol : 0;
                  const heightPct = Math.max(0, (safeVol / maxVol) * 100);

                  return (
                    <div key={idx} className="volume-bar">
                      <div 
                        className="bar-fill"
                        style={{ height: `${heightPct}%` }}
                      />
                      <span className="bar-label">Scan {idx + 1}</span>
                      <span className="bar-value">{safeVol.toFixed(0)}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Clinical Recommendation */}
            <div className="recommendation-card">
              <h3>
                <Info size={20} />
                Clinical Recommendation
              </h3>
              <p>{predictionData.recommendation}</p>
              <div className="disclaimer">
                <strong>Disclaimer:</strong> These predictions are AI-generated and should be 
                used to support, not replace, clinical judgment. Always consult with qualified 
                medical professionals for treatment decisions.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
