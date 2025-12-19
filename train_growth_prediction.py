"""
LSTM Tumor Growth Prediction Training Script
Trains the growth prediction model on historical patient scan data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys

sys.path.append(str(Path(__file__).parent))
from ml_models.growth_prediction.lstm_growth import TumorGrowthLSTM

# ==================== CONFIGURATION ====================
CONFIG = {
    "data_path": "data/growth_prediction/patient_histories.json",  # Will create synthetic if missing
    "output_dir": "ml_models/growth_prediction",
    "model_name": "lstm_growth_model.pth",
    "scaler_name": "growth_scaler.pkl",
    
    # Model hyperparameters
    "input_size": 10,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    
    # Training hyperparameters
    "batch_size": 16,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "sequence_length": 3,  # Minimum scans required
    "prediction_horizon": 1,  # Predict next timestep
    
    # Data split
    "test_size": 0.2,
    "val_size": 0.1,
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# ==================== DATASET CLASS ====================
class GrowthPredictionDataset(Dataset):
    """
    Dataset for tumor growth prediction
    Each sample is a sequence of scans -> next volume
    """
    def __init__(
        self,
        sequences: List[np.ndarray],
        targets: List[float],
        scaler: StandardScaler = None
    ):
        self.sequences = sequences
        self.targets = targets
        self.scaler = scaler
        
        if self.scaler is not None:
            # Normalize sequences
            normalized = []
            for seq in self.sequences:
                seq_reshaped = seq.reshape(-1, seq.shape[-1])
                seq_normalized = self.scaler.transform(seq_reshaped)
                seq_normalized = seq_normalized.reshape(seq.shape)
                normalized.append(seq_normalized)
            self.sequences = normalized
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.FloatTensor([self.targets[idx]])
        return sequence, target


# ==================== DATA GENERATION ====================
def generate_synthetic_patient_data(
    num_patients: int = 50,
    min_scans: int = 4,
    max_scans: int = 12
) -> List[Dict]:
    """
    Generate synthetic patient scan histories for training
    Simulates realistic tumor growth patterns
    """
    patients = []
    
    print(f"Generating synthetic data for {num_patients} patients...")
    
    for patient_id in range(num_patients):
        num_scans = np.random.randint(min_scans, max_scans + 1)
        
        # Initialize random growth parameters
        initial_volume = np.random.uniform(5.0, 50.0)  # cc
        growth_rate = np.random.uniform(-0.05, 0.15)  # -5% to +15% per month
        volatility = np.random.uniform(0.01, 0.05)  # Random fluctuation
        
        scans = []
        current_volume = initial_volume
        
        for scan_idx in range(num_scans):
            # Time progression (monthly scans with some variation)
            scan_date = datetime(2023, 1, 1) + timedelta(days=30 * scan_idx + np.random.randint(-5, 5))
            
            # Volume evolution with noise
            if scan_idx > 0:
                growth = current_volume * (growth_rate + np.random.normal(0, volatility))
                current_volume = max(1.0, current_volume + growth)
            
            # Generate realistic features
            mean_intensity = np.random.uniform(0.3, 0.7)
            std_intensity = np.random.uniform(0.05, 0.15)
            max_diameter = (current_volume * 3 / (4 * np.pi)) ** (1/3) * 2  # Approximate sphere diameter
            surface_area = 4 * np.pi * (max_diameter / 2) ** 2
            compactness = (36 * np.pi * current_volume ** 2) / (surface_area ** 3) if surface_area > 0 else 0
            sphericity = np.random.uniform(0.6, 0.95)
            
            # Tumor location (relatively stable)
            centroid_x = np.random.uniform(0.3, 0.7) + np.random.normal(0, 0.02)
            centroid_y = np.random.uniform(0.3, 0.7) + np.random.normal(0, 0.02)
            centroid_z = np.random.uniform(0.3, 0.7) + np.random.normal(0, 0.02)
            
            scan_data = {
                'scan_date': scan_date.isoformat(),
                'volume': float(current_volume),
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'max_diameter': float(max_diameter),
                'surface_area': float(surface_area),
                'compactness': float(compactness),
                'sphericity': float(sphericity),
                'centroid_x': float(centroid_x),
                'centroid_y': float(centroid_y),
                'centroid_z': float(centroid_z)
            }
            scans.append(scan_data)
        
        patients.append({
            'patient_id': f'PT-SYNTH-{patient_id:04d}',
            'scans': scans,
            'tumor_type': np.random.choice(['Glioblastoma', 'Low-Grade Glioma', 'Meningioma']),
            'growth_pattern': 'progressive' if growth_rate > 0.05 else 'stable'
        })
    
    return patients


def prepare_sequences(
    patients: List[Dict],
    sequence_length: int = 3
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Convert patient scan histories to sequences for training
    
    Args:
        patients: List of patient dictionaries
        sequence_length: Number of past scans to use as input
    
    Returns:
        Tuple of (input_sequences, target_volumes)
    """
    sequences = []
    targets = []
    
    for patient in patients:
        scans = patient['scans']
        
        # Need at least sequence_length + 1 scans
        if len(scans) < sequence_length + 1:
            continue
        
        # Create sliding windows
        for i in range(len(scans) - sequence_length):
            # Extract features for sequence
            sequence = []
            for j in range(i, i + sequence_length):
                scan = scans[j]
                features = [
                    scan['volume'],
                    scan['mean_intensity'],
                    scan['std_intensity'],
                    scan['max_diameter'],
                    scan['surface_area'],
                    scan['compactness'],
                    scan['sphericity'],
                    scan['centroid_x'],
                    scan['centroid_y'],
                    scan['centroid_z']
                ]
                sequence.append(features)
            
            # Target is the next volume
            target_volume = scans[i + sequence_length]['volume']
            
            sequences.append(np.array(sequence, dtype=np.float32))
            targets.append(target_volume)
    
    print(f"Created {len(sequences)} training sequences from {len(patients)} patients")
    return sequences, targets


# ==================== TRAINING FUNCTIONS ====================
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (sequences, targets) in enumerate(dataloader):
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    
    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            mae = torch.abs(outputs - targets).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
    
    return total_loss / len(dataloader), total_mae / len(dataloader)


def plot_training_history(history: Dict, output_path: Path):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training History - Loss')
    ax1.legend()
    ax1.grid(True)
    
    # MAE plot
    ax2.plot(history['val_mae'], label='Val MAE', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE (cc)')
    ax2.set_title('Validation Mean Absolute Error')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Training curves saved to {output_path}")


# ==================== MAIN TRAINING SCRIPT ====================
def main():
    print("=" * 60)
    print("LSTM Tumor Growth Prediction Training")
    print("=" * 60)
    print(f"Device: {CONFIG['device']}")
    print()
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or generate data
    data_path = Path(CONFIG['data_path'])
    
    if data_path.exists():
        print(f"Loading patient data from {data_path}...")
        with open(data_path, 'r') as f:
            patients = json.load(f)
        print(f"Loaded {len(patients)} patients")
    else:
        print("No existing data found. Generating synthetic dataset...")
        patients = generate_synthetic_patient_data(num_patients=50)
        
        # Save synthetic data
        data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(data_path, 'w') as f:
            json.dump(patients, f, indent=2)
        print(f"Synthetic data saved to {data_path}")
    
    print()
    
    # Prepare sequences
    print("Preparing training sequences...")
    sequences, targets = prepare_sequences(patients, CONFIG['sequence_length'])
    
    if len(sequences) == 0:
        print("ERROR: No training sequences created. Need patients with at least 4 scans.")
        return
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        sequences, targets,
        test_size=CONFIG['test_size'],
        random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=CONFIG['val_size'] / (1 - CONFIG['test_size']),
        random_state=42
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Fit scaler on training data
    print("Fitting scaler...")
    all_features = np.vstack([seq.reshape(-1, seq.shape[-1]) for seq in X_train])
    scaler = StandardScaler()
    scaler.fit(all_features)
    
    # Save scaler
    import pickle
    scaler_path = output_dir / CONFIG['scaler_name']
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    print()
    
    # Create datasets
    train_dataset = GrowthPredictionDataset(X_train, y_train, scaler)
    val_dataset = GrowthPredictionDataset(X_val, y_val, scaler)
    test_dataset = GrowthPredictionDataset(X_test, y_test, scaler)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    print("Initializing model...")
    model = TumorGrowthLSTM(
        input_size=CONFIG['input_size'],
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        output_size=1,
        dropout=CONFIG['dropout']
    ).to(CONFIG['device'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Training loop
    print("Starting training...")
    print("-" * 60)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': []
    }
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(CONFIG['num_epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
        
        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, CONFIG['device'])
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{CONFIG['num_epochs']}] | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_mae:.2f} cc")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            model_path = output_dir / CONFIG['model_name']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'config': CONFIG
            }, model_path)
            
            if (epoch + 1) % 10 == 0:
                print(f"  → Best model saved (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print("-" * 60)
    print("Training complete!")
    print()
    
    # Test evaluation
    print("Evaluating on test set...")
    test_loss, test_mae = validate(model, test_loader, criterion, CONFIG['device'])
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.2f} cc")
    print()
    
    # Plot training curves
    plot_path = output_dir / "training_history.png"
    plot_training_history(history, plot_path)
    
    # Save final results
    results = {
        'best_val_loss': float(best_val_loss),
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'num_epochs_trained': len(history['train_loss']),
        'num_train_samples': len(X_train),
        'num_val_samples': len(X_val),
        'num_test_samples': len(X_test),
        'training_date': datetime.now().isoformat()
    }
    
    results_path = output_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    print()
    print("=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"✓ Model: {CONFIG['model_name']}")
    print(f"✓ Test MAE: {test_mae:.2f} cc")
    print(f"✓ Epochs: {len(history['train_loss'])}")
    print(f"✓ Device: {CONFIG['device']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
