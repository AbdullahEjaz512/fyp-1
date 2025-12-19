import api from './api';

export const advancedService = {
  // Growth Prediction
  predictGrowth(payload: any) {
    return api.post('/api/v1/advanced/growth/predict', payload);
  },
  
  getGrowthHistory(patientId: string) {
    return api.get(`/api/v1/advanced/growth/history/${patientId}`);
  },
  
  // Explainability
  explainClassification(payload: any) {
    return api.post('/api/v1/advanced/explain/classification', payload);
  },
  
  explainSegmentation(payload: any) {
    return api.post('/api/v1/advanced/explain/segmentation', payload);
  },
  
  // Visualization
  visualizeSlice(payload: any) {
    return api.post('/api/v1/advanced/visualize/slice', payload);
  },
  
  visualizeMultiView(payload: any) {
    return api.post('/api/v1/advanced/visualize/multiview', payload);
  },
  
  visualizeMontage(payload: any) {
    return api.post('/api/v1/advanced/visualize/montage', payload);
  },
  
  visualize3DProjection(payload: any) {
    return api.post('/api/v1/advanced/visualize/3d-projection', payload);
  },
  
  getVolumeMetrics(fileId: number) {
    return api.get(`/api/v1/advanced/visualize/metrics/${fileId}`);
  },
};
