import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEffect } from 'react';
import { useAuthStore } from './store/authStore';
import { Navbar } from './components/common/Navbar';
import HomePage from './pages/HomePage';
import DashboardPage from './pages/DashboardPage';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import AssistantPage from './pages/AssistantPage';
import VisualizationPage from './pages/VisualizationPage';
import Reconstruction3DPage from './pages/Reconstruction3DPage';
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
          <Navbar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route
                path="/dashboard"
                element={isAuthenticated ? <DashboardPage /> : <Navigate to="/" />}
              />
              <Route
                path="/upload"
                element={isAuthenticated ? <UploadPage /> : <Navigate to="/" />}
              />
              <Route
                path="/results"
                element={isAuthenticated ? <ResultsPage /> : <Navigate to="/" />}
              />
              <Route
                path="/assistant"
                element={isAuthenticated ? <AssistantPage /> : <Navigate to="/" />}
              />
              <Route
                path="/visualization"
                element={isAuthenticated ? <VisualizationPage /> : <Navigate to="/" />}
              />
              <Route
                path="/reconstruction"
                element={isAuthenticated ? <Reconstruction3DPage /> : <Navigate to="/" />}
              />
              <Route path="*" element={<Navigate to="/" />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
