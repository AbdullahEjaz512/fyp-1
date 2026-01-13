import { useState, useEffect } from 'react';
import { X, ChevronRight, ChevronLeft, CheckCircle } from 'lucide-react';
import { useAuthStore } from '../../store/authStore';
import './OnboardingTour.css';

interface TourStep {
  title: string;
  description: string;
  image?: string;
  tips?: string[];
}

const getTourSteps = (isDoctorRole: boolean): TourStep[] => [
  {
    title: 'Welcome to Seg-Mind!',
    description: `Your AI-powered brain tumor analysis platform. Let's take a quick tour to help you get started.`,
    tips: [
      'This tour will only take 1 minute',
      'You can skip or replay it anytime',
      'All features are accessible from the sidebar'
    ]
  },
  {
    title: 'Step 1: Upload Your MRI Scan',
    description: isDoctorRole 
      ? 'Click "Upload" in the sidebar to upload patient MRI scans (NIfTI format). You can then analyze them to get AI-powered insights.'
      : 'Click "Upload" in the sidebar to upload your MRI scans (NIfTI format). Grant access to a doctor who will analyze them for you.',
    tips: isDoctorRole ? [
      'Supported formats: .nii, .nii.gz',
      'Enter patient ID for each upload',
      'Click "Analyze" to run AI analysis'
    ] : [
      'Supported formats: .nii, .nii.gz',
      'Upload multiple scans over time',
      'Click "Grant Access" to share with your doctor'
    ]
  },
  {
    title: 'Step 2: View Results',
    description: 'Go to "Results" to see the AI classification and tumor segmentation. The system shows you exactly what type of tumor was detected and where it\'s located.',
    tips: [
      'Classification: Tumor type with confidence score',
      'Segmentation: 3D tumor boundaries and volume',
      'XAI Heatmap: See what the AI focused on'
    ]
  },
  {
    title: 'Step 3: Visualize in 2D/3D',
    description: 'Use "2D Visualization" to examine individual slices, or "3D Reconstruction" for an interactive 3D tumor model.',
    tips: [
      '2D: Scroll through slices, adjust windowing',
      '3D: Rotate, zoom, and measure the tumor',
      'Export images for reports'
    ]
  },
  {
    title: 'Step 4: Predict Growth',
    description: 'If you have multiple scans over time, use "Growth Prediction" to forecast tumor progression using our LSTM model.',
    tips: [
      'Requires at least 2 historical scans',
      'Predicts future tumor volume',
      'Helps with treatment planning'
    ]
  },
  {
    title: 'You\'re All Set!',
    description: isDoctorRole
      ? 'Start by uploading your first patient scan. Need help? Click the floating assistant button (bottom right) anytime.'
      : 'Start by uploading your first MRI scan. Need help? Click the floating assistant button (bottom right) anytime.',
    tips: isDoctorRole ? [
      'Use the AI assistant for questions',
      'Generate PDF reports from results',
      'Share cases with colleagues for second opinions'
    ] : [
      'Use the AI assistant for questions',
      'Grant access to your doctor to analyze scans',
      'View detailed results and visualizations'
    ]
  }
];

export const OnboardingTour = () => {
  const user = useAuthStore((state) => state.user);
  const isDoctorRole = !!(user?.role && ['doctor', 'radiologist', 'oncologist'].includes(user.role));
  const [isOpen, setIsOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  
  const TOUR_STEPS = getTourSteps(isDoctorRole);

  useEffect(() => {
    // Check if user has seen the tour
    const hasSeenTour = localStorage.getItem('hasSeenOnboarding');
    if (!hasSeenTour) {
      // Show tour after a short delay
      setTimeout(() => setIsOpen(true), 1000);
    }
  }, []);

  const handleComplete = () => {
    localStorage.setItem('hasSeenOnboarding', 'true');
    setIsOpen(false);
    setCurrentStep(0);
  };

  const handleSkip = () => {
    localStorage.setItem('hasSeenOnboarding', 'true');
    setIsOpen(false);
  };

  const handleNext = () => {
    if (currentStep < TOUR_STEPS.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  if (!isOpen) {
    return (
      <button
        className="replay-tour-btn"
        onClick={() => setIsOpen(true)}
        title="Replay Tutorial"
      >
        <CheckCircle size={20} />
        Tutorial
      </button>
    );
  }

  const step = TOUR_STEPS[currentStep];
  const progress = ((currentStep + 1) / TOUR_STEPS.length) * 100;

  return (
    <div className="onboarding-overlay">
      <div className="onboarding-modal">
        <button className="onboarding-close" onClick={handleSkip}>
          <X size={24} />
        </button>

        <div className="onboarding-progress">
          <div className="progress-bar" style={{ width: `${progress}%` }} />
        </div>

        <div className="onboarding-content">
          <div className="step-indicator">
            Step {currentStep + 1} of {TOUR_STEPS.length}
          </div>

          <h2>{step.title}</h2>
          <p className="step-description">{step.description}</p>

          {step.tips && (
            <div className="step-tips">
              <div className="tips-title">💡 Quick Tips:</div>
              <ul>
                {step.tips.map((tip, idx) => (
                  <li key={idx}>{tip}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="onboarding-actions">
          <div className="action-left">
            {currentStep > 0 && (
              <button onClick={handlePrevious} className="btn-outline">
                <ChevronLeft size={18} />
                Previous
              </button>
            )}
          </div>

          <div className="dots-indicator">
            {TOUR_STEPS.map((_, idx) => (
              <div
                key={idx}
                className={`dot ${idx === currentStep ? 'active' : ''} ${idx < currentStep ? 'completed' : ''}`}
              />
            ))}
          </div>

          <div className="action-right">
            {currentStep < TOUR_STEPS.length - 1 ? (
              <button onClick={handleNext} className="btn-primary">
                Next
                <ChevronRight size={18} />
              </button>
            ) : (
              <button onClick={handleComplete} className="btn-primary">
                <CheckCircle size={18} />
                Get Started
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
