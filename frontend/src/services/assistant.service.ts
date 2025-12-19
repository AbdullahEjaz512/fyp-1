import api from './api';

export const assistantService = {
  chat(message: string) {
    return api.post('/api/v1/assistant/chat', { message });
  },
  generateReport(payload: any) {
    return api.post('/api/v1/assistant/report', payload);
  },
  generatePdfReport(payload: any) {
    return api.post('/api/v1/assistant/report/pdf', payload);
  },
  similarCases(caseId: number) {
    return api.get(`/api/v1/assistant/cases/${caseId}/similar`);
  },
};
