import { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import './VisualizationPage.css';
import { advancedService } from '../services/advanced.service';

export default function VisualizationPage() {
  const [searchParams] = useSearchParams();
  const fileId = searchParams.get('fileId') || localStorage.getItem('lastFileId');
  
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
    return (
      <div className="visualization-page">
        <h2>No file selected</h2>
        <p>Please select a case from the Results page.</p>
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
