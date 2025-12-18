"""
Module 5: LSTM-based Tumor Growth Prediction
Predicts future tumor volume based on historical scan sequences
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TumorGrowthLSTM(nn.Module):
    """
    LSTM model for predicting tumor growth trajectory
    Input: Sequence of tumor volumes/features over time
    Output: Predicted future volumes
    """
    def __init__(
        self,
        input_size: int = 10,  # Number of features per timestep
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,  # Predict single volume value
        dropout: float = 0.2
    ):
        super(TumorGrowthLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_size)
        Returns:
            predictions: Tensor of shape (batch, output_size)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Prediction
        output = self.fc(last_hidden)
        
        return output


class GrowthPredictionService:
    """Service for tumor growth prediction and analysis"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TumorGrowthLSTM()
        
        if model_path:
            try:
                self.load_model(model_path)
                logger.info(f"Growth prediction model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Using untrained model.")
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_model(self, path: str):
        """Load trained model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def extract_features(self, scan_data: dict) -> np.ndarray:
        """
        Extract relevant features from scan data
        Features: volume, mean_intensity, std_intensity, max_diameter, 
                  surface_area, compactness, sphericity, location_x/y/z, growth_rate
        """
        features = []
        
        # Volume (cc)
        volume = scan_data.get('volume', 0.0)
        features.append(volume)
        
        # Intensity statistics
        features.append(scan_data.get('mean_intensity', 0.0))
        features.append(scan_data.get('std_intensity', 0.0))
        
        # Morphological features
        features.append(scan_data.get('max_diameter', 0.0))
        features.append(scan_data.get('surface_area', 0.0))
        features.append(scan_data.get('compactness', 0.0))
        features.append(scan_data.get('sphericity', 0.0))
        
        # Location (centroid)
        features.append(scan_data.get('centroid_x', 0.0))
        features.append(scan_data.get('centroid_y', 0.0))
        features.append(scan_data.get('centroid_z', 0.0))
        
        return np.array(features, dtype=np.float32)
    
    def predict_growth(
        self,
        historical_scans: List[dict],
        prediction_steps: int = 3
    ) -> dict:
        """
        Predict future tumor growth based on historical scans
        
        Args:
            historical_scans: List of scan dictionaries with features and timestamps
            prediction_steps: Number of future timesteps to predict
        
        Returns:
            Dictionary with predictions and confidence intervals
        """
        if len(historical_scans) < 2:
            return {
                'error': 'Need at least 2 historical scans for prediction',
                'predictions': [],
                'confidence': 0.0
            }
        
        # Extract features from historical scans
        feature_sequence = []
        for scan in historical_scans:
            features = self.extract_features(scan)
            feature_sequence.append(features)
        
        # Convert to tensor
        sequence = np.array(feature_sequence)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        predictions = []
        confidence_intervals = []
        
        with torch.no_grad():
            # Predict future timesteps
            current_sequence = sequence_tensor
            
            for step in range(prediction_steps):
                # Predict next volume
                pred_volume = self.model(current_sequence)
                pred_value = pred_volume.cpu().numpy()[0, 0]
                
                predictions.append(float(pred_value))
                
                # Estimate confidence interval (simplified)
                # In production, use Monte Carlo dropout or ensemble
                std_dev = abs(pred_value) * 0.15  # 15% uncertainty
                confidence_intervals.append({
                    'lower': float(pred_value - 1.96 * std_dev),
                    'upper': float(pred_value + 1.96 * std_dev)
                })
                
                # Update sequence for next prediction
                # Create new features with predicted volume
                last_features = feature_sequence[-1].copy()
                last_features[0] = pred_value  # Update volume
                
                # Append to sequence
                new_sequence = np.vstack([sequence, last_features])
                current_sequence = torch.FloatTensor(new_sequence).unsqueeze(0).to(self.device)
                
                # Keep only last N timesteps
                if current_sequence.shape[1] > 10:
                    current_sequence = current_sequence[:, -10:, :]
        
        # Calculate growth rate
        if len(historical_scans) >= 2:
            last_volume = historical_scans[-1].get('volume', 0)
            prev_volume = historical_scans[-2].get('volume', 0)
            if prev_volume > 0:
                growth_rate = ((last_volume - prev_volume) / prev_volume) * 100
            else:
                growth_rate = 0.0
        else:
            growth_rate = 0.0
        
        # Determine risk level
        avg_predicted_volume = np.mean(predictions)
        last_actual_volume = historical_scans[-1].get('volume', 0)
        
        if avg_predicted_volume > last_actual_volume * 1.5:
            risk_level = "high"
        elif avg_predicted_volume > last_actual_volume * 1.2:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        return {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'historical_volumes': [s.get('volume', 0) for s in historical_scans],
            'growth_rate': float(growth_rate),
            'risk_level': risk_level,
            'recommendation': self._generate_recommendation(risk_level, growth_rate)
        }
    
    def _generate_recommendation(self, risk_level: str, growth_rate: float) -> str:
        """Generate clinical recommendation based on prediction"""
        if risk_level == "high":
            return (
                "High growth predicted. Recommend immediate follow-up imaging in 1-2 months "
                "and consideration of treatment intensification."
            )
        elif risk_level == "moderate":
            return (
                "Moderate growth predicted. Schedule follow-up imaging in 2-3 months "
                "to monitor progression."
            )
        else:
            return (
                "Stable growth pattern. Continue routine monitoring with follow-up "
                "imaging in 3-6 months as per standard protocol."
            )


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  TUMOR GROWTH PREDICTION - MODULE 5 TEST")
    print("="*70)
    
    # Initialize service
    service = GrowthPredictionService()
    print("\n✓ Growth prediction service initialized")
    
    # Create dummy historical scans
    historical_scans = [
        {
            'volume': 15.2,
            'mean_intensity': 120.5,
            'std_intensity': 25.3,
            'max_diameter': 32.1,
            'surface_area': 450.2,
            'compactness': 0.82,
            'sphericity': 0.75,
            'centroid_x': 120.5,
            'centroid_y': 130.2,
            'centroid_z': 45.8,
            'timestamp': '2024-01-01'
        },
        {
            'volume': 18.4,
            'mean_intensity': 122.1,
            'std_intensity': 26.1,
            'max_diameter': 34.5,
            'surface_area': 485.3,
            'compactness': 0.80,
            'sphericity': 0.73,
            'centroid_x': 121.2,
            'centroid_y': 130.8,
            'centroid_z': 46.1,
            'timestamp': '2024-04-01'
        },
        {
            'volume': 22.1,
            'mean_intensity': 124.3,
            'std_intensity': 27.5,
            'max_diameter': 37.2,
            'surface_area': 520.1,
            'compactness': 0.78,
            'sphericity': 0.71,
            'centroid_x': 122.1,
            'centroid_y': 131.5,
            'centroid_z': 46.5,
            'timestamp': '2024-07-01'
        }
    ]
    
    print(f"\n✓ Testing with {len(historical_scans)} historical scans")
    
    # Predict future growth
    result = service.predict_growth(historical_scans, prediction_steps=3)
    
    print("\n📊 Prediction Results:")
    print(f"   Historical volumes: {result['historical_volumes']}")
    print(f"   Predicted volumes: {result['predictions']}")
    print(f"   Growth rate: {result['growth_rate']:.2f}%")
    print(f"   Risk level: {result['risk_level']}")
    print(f"   Recommendation: {result['recommendation']}")
    
    print("\n" + "="*70)
    print("✓ Module 5 test complete!")
    print("="*70 + "\n")
