import requests
import json

# Login with form data
login_resp = requests.post('http://localhost:8000/api/v1/auth/login', 
    json={'email': 'test_doctor@example.com', 'password': 'password123'})
login_data = login_resp.json()
if 'access_token' not in login_data:
    print(f"Login failed: {login_data}")
    exit(1)
print("Login successful!")
token = login_data['access_token']
headers = {'Authorization': f'Bearer {token}'}

# Get latest result
results = requests.get('http://localhost:8000/api/v1/results/', headers=headers)
if results.ok and results.json():
    latest = results.json()[0]
    print('=== CLASSIFICATION RESULTS ===')
    print(f'Tumor Type: {latest.get("classification_type", "N/A")}')
    print(f'Confidence: {latest.get("classification_confidence", "N/A")}%')
    print(f'WHO Grade: {latest.get("who_grade", "N/A")}')
    print(f'Malignancy: {latest.get("malignancy_level", "N/A")}')
    print(f'Notes: {latest.get("notes", "N/A")}')
    print()
    print('=== SEGMENTATION VOLUMES ===')
    seg = latest.get('segmentation_data', {})
    regions = seg.get('regions', {})
    for region, data in regions.items():
        print(f'{region}: {data.get("volume_mm3", 0):.1f} mm³')
    print(f'Total: {seg.get("total_volume", {}).get("mm3", 0):.1f} mm³')
else:
    print("No results found")
