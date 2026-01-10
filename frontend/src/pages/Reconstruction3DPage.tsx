/**
 * Module 8: 3D Tumor Reconstruction Page
 * Interactive 3D visualization using VTK.js and Three.js
 * Features: Mesh viewing, rotation, zoom, pan, export (STL/OBJ)
 */

import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import './Reconstruction3DPage.css';
import { API_BASE_URL } from '../config/api';

// Import icons
import { Box, Download, Eye, Settings, RefreshCw } from 'lucide-react';

// Will use VTK.js for medical visualization
// Install: npm install @kitware/vtk.js

const Reconstruction3DPage: React.FC = () => {
  const [searchParams] = useSearchParams();
  const fileParam = searchParams.get('fileId') || searchParams.get('file_id') || localStorage.getItem('lastFileId');
  
  // State
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [meshData, setMeshData] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  
  // Viewer state
  const [viewerType, setViewerType] = useState<'vtkjs' | 'threejs'>('vtkjs');
  const [visibleRegions, setVisibleRegions] = useState<{ [key: string]: boolean }>({
    NCR: true,
    ED: true,
    ET: true
  });
  
  // Controls state
  const [wireframe, setWireframe] = useState(false);
  const [backgroundColor, setBackgroundColor] = useState('#1a1a1a');
  const [autoRotate, setAutoRotate] = useState(false);
  const [showInstructions, setShowInstructions] = useState(true);
  
  // Refs
  const vtkContainerRef = useRef<HTMLDivElement>(null);
  const threeContainerRef = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (fileParam) {
      loadMeshData();
      loadStats();
    }
  }, [fileParam]);

  // Reload data when viewer type changes to request correct format (vtkjs vs threejs)
  useEffect(() => {
    if (fileParam) {
      loadMeshData();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewerType]);

  // Re-initialize viewer when interactive settings change (no re-fetch)
  useEffect(() => {
    if (!meshData) return;
    if (viewerType === 'vtkjs') {
      initVTKViewer(meshData);
    } else {
      initThreeJSViewer(meshData);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visibleRegions, wireframe, backgroundColor, autoRotate]);
  
  const loadMeshData = async () => {
    if (!fileParam) return;
    try {
      setLoading(true);
      const token = localStorage.getItem('authToken');
      
      const response = await fetch(
        `${API_BASE_URL}/api/v1/reconstruction/viewer-data/${fileParam}?format=${viewerType}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );
      
      if (!response.ok) {
        throw new Error('Failed to load 3D mesh data');
      }
      
      const data = await response.json();
      setMeshData(data);
      
      // Check if there's an error in the response
      if (data.error) {
        setError(data.error);
        setLoading(false);
        return;
      }
      
      // Check if there are any geometries/regions
      const hasData = (viewerType === 'threejs' && data.geometries?.length > 0) ||
                      (viewerType === 'vtkjs' && data.regions?.length > 0);
      
      if (!hasData) {
        setError('No 3D mesh data available. The segmentation may not contain visible tumor regions.');
        setLoading(false);
        return;
      }
      
      // Initialize viewer based on type
      if (viewerType === 'vtkjs') {
        initVTKViewer(data);
      } else {
        initThreeJSViewer(data);
      }
      
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  const loadStats = async () => {
    if (!fileParam) return;
    try {
      const token = localStorage.getItem('authToken');
      
      const response = await fetch(
        `${API_BASE_URL}/api/v1/reconstruction/stats/${fileParam}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );
      
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (err) {
      console.error('Failed to load stats:', err);
    }
  };
  
  const initVTKViewer = async (data: any) => {
    if (!vtkContainerRef.current) return;
    
    try {
      // Clear any previous canvas to ensure fresh render
      vtkContainerRef.current.innerHTML = '';
      // Dynamically import VTK.js (only when needed)
      // @ts-ignore VTK types not bundled
      const vtk = await import('@kitware/vtk.js');
      const vtkFullScreenRenderWindow = await import('@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow');
      const vtkActor = await import('@kitware/vtk.js/Rendering/Core/Actor');
      const vtkMapper = await import('@kitware/vtk.js/Rendering/Core/Mapper');
      const vtkPolyData = await import('@kitware/vtk.js/Common/DataModel/PolyData');
      
      // Create full screen render window
      const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
        container: vtkContainerRef.current,
        background: [0.1, 0.1, 0.1]
      });
      
      const renderer = fullScreenRenderer.getRenderer();
      const renderWindow = fullScreenRenderer.getRenderWindow();
      
      // Configure interactor for proper mouse controls
      const interactor = renderWindow.getInteractor();
      interactor.setDesiredUpdateRate(15.0);
      interactor.setStillUpdateRate(0.001);
      
      // Add each region as separate actor
      data.regions?.forEach((region: any) => {
        const regionKey = region?.metadata?.region || region?.name;
        if (!visibleRegions[regionKey]) return;
        
        // Create polydata
        const polydata = vtkPolyData.newInstance();
        
        // Set points
        const points = new Float32Array(region.points.values);
        polydata.getPoints().setData(points, 3);
        
        // Set polygons
        const polys = new Uint32Array(region.polys.values);
        polydata.getPolys().setData(polys);
        
        // Create mapper
        const mapper = vtkMapper.newInstance();
        mapper.setInputData(polydata);
        
        // Create actor
        const actor = vtkActor.newInstance();
        actor.setMapper(mapper);
        
        // Set color
        const color = region.metadata.color.map((c: number) => c / 255);
        actor.getProperty().setColor(color[0], color[1], color[2]);
        actor.getProperty().setOpacity(region.metadata.opacity);
        
        if (wireframe) {
          actor.getProperty().setRepresentationToWireframe();
        }
        
        renderer.addActor(actor);
      });
      
      renderer.resetCamera();
      renderWindow.render();
      
    } catch (err) {
      console.error('VTK.js initialization failed:', err);
      setError('VTK.js viewer initialization failed. Install: npm install @kitware/vtk.js');
    }
  };
  
  const initThreeJSViewer = async (data: any) => {
    if (!threeContainerRef.current) return;
    
    // Wait for container to have valid dimensions
    const container = threeContainerRef.current;
    if (container.clientWidth === 0 || container.clientHeight === 0) {
      console.warn('Container has zero dimensions, retrying in 100ms...');
      setTimeout(() => initThreeJSViewer(data), 100);
      return;
    }
    
    try {
      // Dynamically import Three.js
      const THREE = await import('three');
      // @ts-ignore three type import
      const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls');
      
      const width = container.clientWidth;
      const height = container.clientHeight;
      
      console.log('Initializing Three.js with dimensions:', width, 'x', height);
      
      // Create scene
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(backgroundColor);
      
      // Create camera
      const camera = new THREE.PerspectiveCamera(
        75,
        width / height,
        0.1,
        10000
      );
      camera.position.set(200, 200, 200);
      
      // Create renderer
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setPixelRatio((window as any).devicePixelRatio || 1);
      renderer.setSize(width, height);
      container.innerHTML = '';
      container.appendChild(renderer.domElement);
      
      // Add controls
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.autoRotate = autoRotate;
      controls.autoRotateSpeed = 2.0;
      controls.enablePan = true;
      controls.enableZoom = true;
      (controls as any).mouseButtons = {
        LEFT: (THREE as any).MOUSE.ROTATE,
        MIDDLE: (THREE as any).MOUSE.PAN,
        RIGHT: (THREE as any).MOUSE.PAN
      };
      
      // Add lights
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
      scene.add(ambientLight);
      
      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(100, 100, 100);
      scene.add(directionalLight);
      
      // Add each geometry
      data.geometries?.forEach((geom: any) => {
        const regionName = geom.metadata.region;
        if (!visibleRegions[regionName]) return;
        
        console.log('Loading region:', regionName, 'with', geom.data.attributes.position.array.length / 3, 'vertices');
        
        // Create geometry
        const geometry = new THREE.BufferGeometry();
        
        const positions = new Float32Array(geom.data.attributes.position.array);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        if (geom.data.index && geom.data.index.array) {
          const indices = new Uint32Array(geom.data.index.array);
          geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        }
        
        geometry.computeVertexNormals();
        geometry.computeBoundingSphere();
        
        // Create material
        const color = new THREE.Color(
          geom.metadata.color[0] / 255,
          geom.metadata.color[1] / 255,
          geom.metadata.color[2] / 255
        );
        
        const material = new THREE.MeshPhongMaterial({
          color: color,
          opacity: geom.metadata.opacity,
          transparent: true,
          wireframe: wireframe,
          side: THREE.DoubleSide
        });
        
        // Create mesh
        const mesh = new THREE.Mesh(geometry, material);
        scene.add(mesh);
      });
      
      // Adjust camera to fit all objects
      const box = new THREE.Box3().setFromObject(scene);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      const fov = camera.fov * (Math.PI / 180);
      let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
      cameraZ *= 1.5; // Add some padding
      
      camera.position.set(center.x + cameraZ, center.y + cameraZ, center.z + cameraZ);
      camera.lookAt(center);
      controls.target.copy(center);
      controls.update();
      
      console.log('Scene bounds:', box, 'Camera position:', camera.position);
      
      // Animation loop
      const animate = () => {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      };
      animate();
      
      // Handle resize
      const handleResize = () => {
        if (!container) return;
        const newWidth = container.clientWidth;
        const newHeight = container.clientHeight;
        
        if (newWidth === 0 || newHeight === 0) return;
        
        camera.aspect = newWidth / newHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(newWidth, newHeight);
      };
      window.addEventListener('resize', handleResize);
      
    } catch (err) {
      console.error('Three.js initialization failed:', err);
      setError('Three.js viewer initialization failed. Install: npm install three');
    }
  };
  
  const toggleRegionVisibility = (region: string) => {
    setVisibleRegions(prev => ({
      ...prev,
      [region]: !prev[region]
    }));
  };
  
  const downloadSTL = async (region: string) => {
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(
        `${API_BASE_URL}/api/v1/reconstruction/export/stl/${fileParam}/${region}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );
      
      if (!response.ok) throw new Error('STL download failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `tumor_${region}.stl`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      alert('Failed to download STL: ' + err.message);
    }
  };
  
  const downloadOBJ = async (region: string) => {
    try {
      const token = localStorage.getItem('authToken');
      const response = await fetch(
        `${API_BASE_URL}/api/v1/reconstruction/export/obj/${fileParam}/${region}`,
        {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );
      
      if (!response.ok) throw new Error('OBJ download failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `tumor_${region}.obj`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      alert('Failed to download OBJ: ' + err.message);
    }
  };
  
  if (!fileParam) {
    return (
      <div className="reconstruction-page">
        <div className="error-message">
          <p>No file ID provided</p>
        </div>
      </div>
    );
  }
  
  return (
    <div className="reconstruction-page">
      <div className="reconstruction-header">
        <h1><Box size={32} /> 3D Tumor Reconstruction</h1>
        <p>Interactive 3D visualization with VTK.js and Three.js</p>
      </div>
      
      {error && (
        <div className="error-banner">
          <p>{error}</p>
          <button onClick={() => loadMeshData()}>
            <RefreshCw size={16} /> Retry
          </button>
        </div>
      )}
      
      <div className="reconstruction-container">
        {/* Sidebar Controls */}
        <div className="reconstruction-sidebar">
          <div className="control-section">
            <h3><Settings size={20} /> Viewer Settings</h3>
            
            <div className="control-group">
              <label>Viewer Type:</label>
              <select 
                value={viewerType} 
                onChange={(e) => {
                  // Only set state; a useEffect will refetch with correct format
                  setViewerType(e.target.value as 'vtkjs' | 'threejs');
                }}
              >
                <option value="vtkjs">VTK.js (Medical)</option>
                <option value="threejs">Three.js (Interactive)</option>
              </select>
            </div>
            
            <div className="control-group">
              <label>
                <input
                  type="checkbox"
                  checked={wireframe}
                  onChange={(e) => {
                    setWireframe(e.target.checked);
                    if (meshData) {
                      if (viewerType === 'vtkjs') initVTKViewer(meshData);
                      else initThreeJSViewer(meshData);
                    }
                  }}
                />
                Wireframe Mode
              </label>
            </div>
            
            {viewerType === 'threejs' && (
              <div className="control-group">
                <label>
                  <input
                    type="checkbox"
                    checked={autoRotate}
                    onChange={(e) => {
                      setAutoRotate(e.target.checked);
                      if (meshData) initThreeJSViewer(meshData);
                    }}
                  />
                  Auto Rotate
                </label>
              </div>
            )}
            
            <div className="control-group">
              <label>Background:</label>
              <input
                type="color"
                value={backgroundColor}
                onChange={(e) => {
                  setBackgroundColor(e.target.value);
                  if (meshData && viewerType === 'threejs') {
                    initThreeJSViewer(meshData);
                  }
                }}
              />
            </div>
          </div>
          
          <div className="control-section">
            <h3><Eye size={20} /> Visible Regions</h3>
            
            {['NCR', 'ED', 'ET'].map(region => (
              <div key={region} className="region-toggle">
                <label>
                  <input
                    type="checkbox"
                    checked={visibleRegions[region]}
                    onChange={() => toggleRegionVisibility(region)}
                  />
                  <span className={`region-color ${region.toLowerCase()}`}></span>
                  {region === 'NCR' && 'Necrotic Core'}
                  {region === 'ED' && 'Edema'}
                  {region === 'ET' && 'Enhancing Tumor'}
                </label>
              </div>
            ))}
          </div>
          
          <div className="control-section">
            <h3><Download size={20} /> Export</h3>
            
            {['NCR', 'ED', 'ET'].map(region => (
              <div key={region} className="export-buttons">
                <span>{region}:</span>
                <button onClick={() => downloadSTL(region)}>STL</button>
                <button onClick={() => downloadOBJ(region)}>OBJ</button>
              </div>
            ))}
          </div>
          
          {stats && (
            <div className="control-section">
              <h3>Statistics</h3>
              <div className="stats-list">
                <div className="stat-item">
                  <span>Total Vertices:</span>
                  <strong>{stats.total_vertices?.toLocaleString()}</strong>
                </div>
                <div className="stat-item">
                  <span>Total Faces:</span>
                  <strong>{stats.total_faces?.toLocaleString()}</strong>
                </div>
                <div className="stat-item">
                  <span>Surface Area:</span>
                  <strong>{stats.total_surface_area_mm2?.toFixed(2)} mm²</strong>
                </div>
                <div className="stat-item">
                  <span>Regions:</span>
                  <strong>{stats.num_regions}</strong>
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* 3D Viewer */}
        <div className="reconstruction-viewer">
          {loading && (
            <div className="viewer-loading">
              <div className="spinner"></div>
              <p>Loading 3D mesh...</p>
            </div>
          )}
          
          {viewerType === 'vtkjs' && (
            <div ref={vtkContainerRef} className="vtk-container"></div>
          )}
          
          {viewerType === 'threejs' && (
            <div ref={threeContainerRef} className="threejs-container"></div>
          )}
          
          {showInstructions && (
            <div className="viewer-instructions" style={{ 
              position: 'absolute', 
              bottom: '20px', 
              left: '20px', 
              maxWidth: '350px',
              maxHeight: '40vh',
              overflowY: 'auto',
              backgroundColor: 'rgba(0, 0, 0, 0.85)',
              backdropFilter: 'blur(10px)',
              padding: '15px',
              borderRadius: '8px',
              border: '1px solid rgba(0, 255, 255, 0.3)',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <p style={{ margin: 0, fontSize: '1em', fontWeight: 'bold', color: '#00ffff' }}>Controls & Clinical Info</p>
                <button 
                  onClick={() => setShowInstructions(false)}
                  style={{
                    background: 'rgba(255, 255, 255, 0.1)',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    color: '#fff',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontSize: '0.85em'
                  }}
                >
                  Hide
                </button>
              </div>
              
              <div style={{ fontSize: '0.9em' }}>
                <p style={{ margin: '8px 0', fontWeight: 'bold' }}>Mouse Controls:</p>
                <p style={{ margin: '4px 0', fontSize: '0.85em' }}>• Left Click + Drag: Rotate</p>
                <p style={{ margin: '4px 0', fontSize: '0.85em' }}>• Right Click: Pan (VTK) / Middle Mouse (Three.js)</p>
                <p style={{ margin: '4px 0', fontSize: '0.85em' }}>• Scroll: Zoom</p>
                
                <div style={{ marginTop: '12px', paddingTop: '12px', borderTop: '1px solid rgba(255,255,255,0.2)' }}>
                  <p style={{ margin: '8px 0', fontWeight: 'bold' }}>Clinical Utility:</p>
                  <ul style={{ fontSize: '0.8em', margin: '8px 0 0 15px', paddingLeft: 0, lineHeight: '1.5' }}>
                    <li style={{ marginBottom: '6px' }}><strong>Surgical Planning:</strong> Visualize tumor location and spatial relationships</li>
                    <li style={{ marginBottom: '6px' }}><strong>Risk Assessment:</strong> Identify critical regions to minimize surgical risks</li>
                    <li style={{ marginBottom: '6px' }}><strong>Patient Communication:</strong> Show tumor anatomy clearly</li>
                    <li style={{ marginBottom: '6px' }}><strong>Progress Monitoring:</strong> Track tumor changes over time</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
          
          {!showInstructions && (
            <button 
              onClick={() => setShowInstructions(true)}
              style={{
                position: 'absolute',
                bottom: '20px',
                left: '20px',
                background: 'rgba(0, 255, 255, 0.2)',
                border: '1px solid rgba(0, 255, 255, 0.5)',
                color: '#00ffff',
                padding: '10px 16px',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '0.9em',
                fontWeight: 'bold',
                backdropFilter: 'blur(5px)'
              }}
            >
              Show Instructions
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default Reconstruction3DPage;
