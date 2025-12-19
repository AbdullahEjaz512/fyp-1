# Kaggle LSTM Growth Prediction Training Notebook
# Upload this to Kaggle and enable GPU accelerator

## Cell 1: Setup and Imports

```python
# Install additional packages if needed
!pip install scikit-learn matplotlib -q

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")
```

## Cell 2: Define LSTM Model

```python
class TumorGrowthLSTM(nn.Module):
    """LSTM model for predicting tumor growth"""
    
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 128,  # Larger for production
        num_layers: int = 3,     # Deeper
        output_size: int = 1,
        dropout: float = 0.3
    ):
        super(TumorGrowthLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        predictions = self.fc(last_output)
        return predictions

# Test model
model = TumorGrowthLSTM(input_size=10, hidden_size=128, num_layers=3).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## Cell 3: Dataset Class

```python
class GrowthPredictionDataset(Dataset):
    """Dataset for tumor growth prediction"""
    
    def __init__(self, sequences, targets, scaler=None):
        self.sequences = sequences
        self.targets = targets
        self.scaler = scaler
        
        if self.scaler is not None:
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

print("Dataset class defined ✓")
```

## Cell 4: Load Data

```python
# Upload your patient_histories.json file to Kaggle
# Or use the Kaggle dataset if available

# IMPORTANT: Kaggle converts underscores to HYPHENS in paths!
# Use: patient-histories (with hyphen, not underscore)
DATA_PATH = "/kaggle/input/patient-histories/patient_histories.json"

with open(DATA_PATH, 'r') as f:
    patients = json.load(f)

print(f"Loaded {len(patients)} patients")

# Prepare sequences
def prepare_sequences(patients, sequence_length=3):
    sequences = []
    targets = []
    
    for patient in patients:
        scans = patient['scans']
        
        if len(scans) < sequence_length + 1:
            continue
        
        for i in range(len(scans) - sequence_length):
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
            
            target_volume = scans[i + sequence_length]['volume']
            
            sequences.append(np.array(sequence, dtype=np.float32))
            targets.append(target_volume)
    
    return sequences, targets

sequences, targets = prepare_sequences(patients, sequence_length=4)  # Increased from 3 for more context
print(f"Created {len(sequences)} training sequences")
```

## Cell 5: Split and Scale Data

```python
# Split data
X_temp, X_test, y_temp, y_test = train_test_split(
    sequences, targets, test_size=0.15, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Fit scaler
all_features = np.vstack([seq.reshape(-1, seq.shape[-1]) for seq in X_train])
scaler = StandardScaler()
scaler.fit(all_features)

# Create datasets
train_dataset = GrowthPredictionDataset(X_train, y_train, scaler)
val_dataset = GrowthPredictionDataset(X_val, y_val, scaler)
test_dataset = GrowthPredictionDataset(X_test, y_test, scaler)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print("Data prepared ✓")
```

## Cell 6: Training Functions

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for sequences, targets in dataloader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
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

print("Training functions defined ✓")
```

## Cell 7: Train Model

```python
# Initialize model
model = TumorGrowthLSTM(
    input_size=10,
    hidden_size=256,  # Increased from 128 for more capacity
    num_layers=3,
    dropout=0.3
).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Lower LR, more regularization
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=25  # Wait longer before reducing LR
)

# Training loop
num_epochs = 200
best_val_loss = float('inf')
patience = 60  # Increased from 30 to let model train longer
patience_counter = 0

history = {'train_loss': [], 'val_loss': [], 'val_mae': []}

print("Starting training...")
print("-" * 60)

for epoch in range(num_epochs):
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_mae = validate(model, val_loader, criterion, device)
    
    # Update scheduler
    scheduler.step(val_loss)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_mae'].append(val_mae)
    
    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val MAE: {val_mae:.2f} cc")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_mae': val_mae
        }, '/kaggle/working/lstm_growth_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"  → Best model saved (Val MAE: {val_mae:.2f} cc)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

print("-" * 60)
print("Training complete!")
```

## Cell 8: Evaluate on Test Set

```python
# Load best model
checkpoint = torch.load('/kaggle/working/lstm_growth_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Test
test_loss, test_mae = validate(model, test_loader, criterion, device)

print("=" * 60)
print("TEST RESULTS")
print("=" * 60)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.2f} cc")
print(f"Test RMSE: {np.sqrt(test_loss):.2f} cc")
print("=" * 60)
```

## Cell 9: Plot Training History

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# Loss plot
ax1.plot(history['train_loss'], label='Train Loss', alpha=0.7)
ax1.plot(history['val_loss'], label='Val Loss', alpha=0.7)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.set_title('Training History - Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# MAE plot
ax2.plot(history['val_mae'], label='Val MAE', color='orange', alpha=0.7)
ax2.axhline(y=0.8, color='r', linestyle='--', label='Target (0.8 cc)', alpha=0.5)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MAE (cc)')
ax2.set_title('Validation Mean Absolute Error')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/training_history.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFinal Val MAE: {history['val_mae'][-1]:.2f} cc")
print(f"Target: < 0.8 cc")
print(f"Status: {'✓ ACHIEVED' if history['val_mae'][-1] < 0.8 else '✗ NOT ACHIEVED'}")
```

## Cell 10: Save Scaler

```python
# Save scaler
with open('/kaggle/working/growth_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Scaler saved ✓")
print("\nDownload these files:")
print("1. lstm_growth_model.pth")
print("2. growth_scaler.pkl")
print("3. training_history.png")
```

## Cell 11: Sample Predictions

```python
# Get some test samples
model.eval()
samples = []

with torch.no_grad():
    for i, (sequences, targets) in enumerate(test_loader):
        if i >= 5:  # Get 5 samples
            break
        
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        outputs = model(sequences)
        
        for j in range(min(3, len(sequences))):
            samples.append({
                'predicted': outputs[j].cpu().item(),
                'actual': targets[j].cpu().item(),
                'error': abs(outputs[j].cpu().item() - targets[j].cpu().item())
            })

print("Sample Predictions:")
print("-" * 60)
for i, sample in enumerate(samples[:10], 1):
    print(f"{i:2d}. Predicted: {sample['predicted']:6.2f} cc | "
          f"Actual: {sample['actual']:6.2f} cc | "
          f"Error: {sample['error']:5.2f} cc")
print("-" * 60)
print(f"Average Error: {np.mean([s['error'] for s in samples]):.2f} cc")
```

---

## Instructions for Kaggle:

1. **Create new notebook** on Kaggle
2. **Enable GPU**: Settings → Accelerator → GPU T4 x2
3. **Upload data**: 
   - Upload `patient_histories.json` to Input
   - Or add as dataset
4. **Copy cells** from above into notebook
5. **Run all cells** (takes ~1-2 hours)
6. **Download outputs**:
   - `lstm_growth_model.pth`
   - `growth_scaler.pkl`
   - `training_history.png`
7. **Place in project**:
   - Copy to `ml_models/growth_prediction/`

---

## Expected Results:

- **Train MAE**: < 0.3 cc
- **Val MAE**: < 0.5 cc
- **Test MAE**: < 0.8 cc (production target)
- **Training time**: 1-2 hours on GPU

---

## Next Steps After Training:

1. Download model files
2. Place in `ml_models/growth_prediction/`
3. Test with: `python test_advanced_modules.py`
4. Verify predictions are realistic
5. Deploy to production
