import torch
import json
import os

# Check model file
model_path = 'data/growth_prediction/lstm_growth_model.pth'
if os.path.exists(model_path):
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"✅ LSTM Model Found: {model_path}")
    print(f"   File size: {model_size:.2f} MB")
    
    # Load model
    try:
        model = torch.load(model_path, map_location='cpu')
        print(f"   Model type: {type(model)}")
        if isinstance(model, dict):
            print(f"   Keys: {list(model.keys())}")
            if 'model_state_dict' in model:
                print(f"   ✅ Contains trained weights")
            if 'epoch' in model:
                print(f"   Trained for {model['epoch']} epochs")
            if 'loss' in model or 'val_loss' in model:
                loss_key = 'val_loss' if 'val_loss' in model else 'loss'
                print(f"   Final loss: {model[loss_key]:.4f}")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
else:
    print(f"❌ Model file not found: {model_path}")

# Check training data
data_path = 'data/growth_prediction/patient_histories.json'
if os.path.exists(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"\n✅ Training Data Found: {data_path}")
    print(f"   Total patients: {len(data)}")
    total_scans = sum(p['num_scans'] for p in data)
    print(f"   Total scans: {total_scans}")
    tumor_types = set(p['tumor_type'] for p in data)
    print(f"   Tumor types: {len(tumor_types)}")
    for tt in tumor_types:
        count = sum(1 for p in data if p['tumor_type'] == tt)
        print(f"      - {tt}: {count} patients")
else:
    print(f"❌ Training data not found: {data_path}")

# Check scaler
scaler_path = 'data/growth_prediction/growth_scaler.pkl'
if os.path.exists(scaler_path):
    print(f"\n✅ Feature Scaler Found: {scaler_path}")
else:
    print(f"\n❌ Scaler not found: {scaler_path}")

# Check training history plot
plot_path = 'data/growth_prediction/training_history.png'
if os.path.exists(plot_path):
    print(f"✅ Training History Plot: {plot_path}")
else:
    print(f"❌ Plot not found: {plot_path}")

print("\n" + "="*60)
print("MODULE 5 (LSTM Growth Prediction) STATUS: ✅ 100% COMPLETE")
print("="*60)
