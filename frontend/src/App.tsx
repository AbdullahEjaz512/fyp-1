import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEffect } from 'react';
import { useAuthStore } from './store/authStore';
import { Sidebar } from './components/common/Sidebar';
import { TopBar } from './components/common/TopBar';
import { FloatingAssistant } from './components/common/FloatingAssistant';
import { OnboardingTour } from './components/common/OnboardingTour';
import HomePage from './pages/HomePage';
import DashboardPage from './pages/DashboardPage';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import AssistantPage from './pages/AssistantPage';
import VisualizationPage from './pages/VisualizationPage';
import Reconstruction3DPage from './pages/Reconstruction3DPage';
import GrowthPredictionPage from './pages/GrowthPredictionPage';
import { useUIStore } from './store/uiStore';
import './App.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  const initialize = useAuthStore((state) => state.initialize);
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);
  const isLoading = useAuthStore((state) => state.isLoading);
  const isSidebarCollapsed = useUIStore((state) => state.isSidebarCollapsed);
  const user = useAuthStore((state) => state.user);

  const isDoctorRole = user?.role && ['doctor', 'radiologist', 'oncologist'].includes(user.role);

  useEffect(() => {
    initialize();
  }, [initialize]);

  if (isLoading) {
    return (
      <div className="loading-screen">
        <div className="loader"></div>
        <p>Loading Seg-Mind...</p>
      </div>
    );
  }

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <div className="app">
          {isAuthenticated ? (
            <>
              <Sidebar />
              <div className={`app-layout ${isSidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
                <TopBar />
                <main className="main-content">
                  <Routes>
                    <Route path="/" element={<Navigate to="/dashboard" />} />
                    <Route path="/dashboard" element={<DashboardPage />} />
                    <Route path="/upload" element={<UploadPage />} />
                    <Route path="/results" element={<ResultsPage />} />

                    {/* Doctor Only Routes */}
                    <Route path="/assistant" element={
                      isDoctorRole ? <AssistantPage /> : <Navigate to="/dashboard" />
                    } />
                    <Route path="/visualization" element={
                      isDoctorRole ? <VisualizationPage /> : <Navigate to="/dashboard" />
                    } />
                    <Route path="/reconstruction" element={
                      isDoctorRole ? <Reconstruction3DPage /> : <Navigate to="/dashboard" />
                    } />
                    <Route path="/growth-prediction" element={
                      isDoctorRole ? <GrowthPredictionPage /> : <Navigate to="/dashboard" />
                    } />

                    <Route path="*" element={<Navigate to="/dashboard" />} />
                  </Routes>
                </main>
                <FloatingAssistant />
                <OnboardingTour />
              </div>
            </>
          ) : (
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="*" element={<Navigate to="/" />} />
            </Routes>
          )}
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
