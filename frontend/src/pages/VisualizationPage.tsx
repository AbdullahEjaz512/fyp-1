import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { fileService } from '../services/file.service';
import './VisualizationPage.css';
import { advancedService } from '../services/advanced.service';
import { FileText, Calendar, Eye } from 'lucide-react';

export default function VisualizationPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const fileId = searchParams.get('fileId') || localStorage.getItem('lastFileId');
  
  // Query for file list when no file is selected
  const { data: files } = useQuery({
    queryKey: ['files'],
    queryFn: fileService.listFiles,
    enabled: !fileId,
  });
  
  const [view, setView] = useState<'slice' | 'multiview' | 'montage' | '3d'>('multiview');
  const [sliceIdx, setSliceIdx] = useState(64);
  const [axis, setAxis] = useState(2);
  const [includeSegmentation, setIncludeSegmentation] = useState(true);
  const [loading, setLoading] = useState(false);
  const [imageData, setImageData] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<any>(null);

  const loadVisualization = async () => {
    if (!fileId) return;
    
    setLoading(true);
    try {
      let response;
      
      switch (view) {
        case 'slice':
          response = await advancedService.visualizeSlice({
            file_id: parseInt(fileId),
            slice_idx: sliceIdx,
            axis,
            include_segmentation: includeSegmentation
          });
          break;
        
        case 'multiview':
          response = await advancedService.visualizeMultiView({
            file_id: parseInt(fileId),
            include_segmentation: includeSegmentation
          });
          break;
        
        case 'montage':
          response = await advancedService.visualizeMontage({
            file_id: parseInt(fileId),
            num_slices: 12,
            axis
          });
          break;
        
        case '3d':
          response = await advancedService.visualize3DProjection({
            file_id: parseInt(fileId),
            method: 'mip'
          });
          break;
      }
      
      setImageData(response.data.image_base64);
    } catch (err) {
      console.error('Visualization error:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadMetrics = async () => {
    if (!fileId) return;
    
    try {
      const response = await advancedService.getVolumeMetrics(parseInt(fileId));
      setMetrics(response.data.metrics);
    } catch (err) {
      console.error('Metrics error:', err);
    }
  };

  useEffect(() => {
    loadVisualization();
    loadMetrics();
  }, [fileId, view, sliceIdx, axis, includeSegmentation]);

  if (!fileId) {
    const analyzedFiles = files?.filter(f => f.status === 'analyzed') || [];
    return (
      <div className="visualization-page">
        <div className="container" style={{ maxWidth: '1200px', margin: '0 auto', padding: '2rem' }}>
          <div className="page-header" style={{ textAlign: 'center', marginBottom: '2rem' }}>
            <h1 style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', justifyContent: 'center', fontSize: '2rem' }}>
              <Eye size={32} />
              2D Visualization
            </h1>
            <p style={{ color: '#64748b' }}>Select a file to view 2D slices</p>
          </div>
          
          {analyzedFiles.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '3rem' }}>
              <p>No analyzed files found. Please upload and analyze a scan first.</p>
              <button onClick={() => navigate('/upload')} className="btn btn-primary" style={{ marginTop: '1rem' }}>
                Go to Upload
              </button>
            </div>
          ) : (
            <div style={{ maxWidth: '800px', margin: '0 auto' }}>
              <h3 style={{ marginBottom: '1.5rem' }}>Select a Scan:</h3>
              <div style={{ display: 'grid', gap: '1rem' }}>
                {analyzedFiles.map((file: any) => (
                  <div
                    key={file.file_id}
                    onClick={() => navigate(`/visualization?fileId=${file.file_id}`)}
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
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      <FileText size={24} color="#667eea" />
                      <div style={{ flex: 1 }}>
                        <div style={{ fontWeight: 600, marginBottom: '0.25rem' }}>{file.filename}</div>
                        <div style={{ fontSize: '0.875rem', color: '#64748b', display: 'flex', gap: '1rem' }}>
                          <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                            <Calendar size={14} />
                            {new Date(file.upload_date).toLocaleDateString()}
                          </span>
                          {file.patient_id && <span>Patient: {file.patient_id}</span>}
                        </div>
                      </div>
                      <Eye size={20} color="#667eea" />
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

  return (
    <div className="visualization-page">
      <div className="viz-header">
        <h2>MRI Visualization</h2>
        <p>File ID: {fileId}</p>
      </div>

      <div className="viz-controls">
        <div className="control-group">
          <label>View Type:</label>
          <select value={view} onChange={(e) => setView(e.target.value as any)}>
            <option value="multiview">Multi-View</option>
            <option value="slice">Single Slice</option>
            <option value="montage">Montage</option>
            <option value="3d">3D Projection</option>
          </select>
        </div>

        {view === 'slice' && (
          <>
            <div className="control-group">
              <label>Slice Index:</label>
              <input
                type="range"
                min="0"
                max="128"
                value={sliceIdx}
                onChange={(e) => setSliceIdx(parseInt(e.target.value))}
              />
              <span>{sliceIdx}</span>
            </div>
            <div className="control-group">
              <label>Axis:</label>
              <select value={axis} onChange={(e) => setAxis(parseInt(e.target.value))}>
                <option value="0">Sagittal</option>
                <option value="1">Coronal</option>
                <option value="2">Axial</option>
              </select>
            </div>
          </>
        )}

        <div className="control-group">
          <label>
            <input
              type="checkbox"
              checked={includeSegmentation}
              onChange={(e) => setIncludeSegmentation(e.target.checked)}
            />
            Show Segmentation Overlay
          </label>
        </div>

        <button onClick={loadVisualization} disabled={loading}>
          {loading ? 'Loading...' : 'Refresh'}
        </button>
      </div>

      <div className="viz-content">
        {loading && <div className="loader">Loading visualization...</div>}
        {imageData && !loading && (
          <img
            src={`data:image/png;base64,${imageData}`}
            alt="MRI Visualization"
            className="viz-image"
          />
        )}
      </div>

      {metrics && (
        <div className="viz-metrics">
          <h3>Volume Metrics</h3>
          <div className="metrics-grid">
            {Object.entries(metrics).map(([region, data]: [string, any]) => (
              <div key={region} className="metric-card">
                <h4>{region}</h4>
                <p>Volume: {data.volume_cc?.toFixed(2)} cc</p>
                <p>Voxels: {data.voxels?.toLocaleString()}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
