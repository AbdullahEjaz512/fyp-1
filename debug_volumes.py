"""Debug script to check volume calculations"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000/api/v1"
TEST_EMAIL = "test_doctor@example.com"
TEST_PASSWORD = "password123"

def login():
    login_data = {"email": TEST_EMAIL, "password": TEST_PASSWORD}
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    if response.status_code == 200:
        return response.json()["access_token"]
    return None

def get_results(token, file_id):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/analyze/results/{file_id}", headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

def main():
    token = login()
    if not token:
        print("Login failed")
        return
    
    # Get results for the latest file (adjust ID as needed)
    file_id = 57  # Change this to your file ID
    results = get_results(token, file_id)
    
    if results and results.get("analyses"):
        analysis = results["analyses"][0]
        seg = analysis.get("segmentation", {})
        
        print("=== Segmentation Data ===")
        print(json.dumps(seg, indent=2))
        
        # Calculate what total should be
        regions = seg.get("regions", {})
        ncr = regions.get("NCR", {})
        ed = regions.get("ED", {})
        et = regions.get("ET", {})
        
        print("\n=== Individual Region Volumes ===")
        print(f"NCR: {ncr.get('volume_mm3', 0)} mm³, {ncr.get('voxel_count', ncr.get('volume_voxels', 0))} voxels")
        print(f"ED: {ed.get('volume_mm3', 0)} mm³, {ed.get('voxel_count', ed.get('volume_voxels', 0))} voxels")
        print(f"ET: {et.get('volume_mm3', 0)} mm³, {et.get('voxel_count', et.get('volume_voxels', 0))} voxels")
        
        total = seg.get("total_volume", {})
        print(f"\n=== Total Volume (from API) ===")
        print(f"Total: {total.get('mm3', 0)} mm³, {total.get('voxels', 0)} voxels")
        
        # What it should be
        calculated_mm3 = ncr.get('volume_mm3', 0) + ed.get('volume_mm3', 0) + et.get('volume_mm3', 0)
        calculated_voxels = (ncr.get('voxel_count', ncr.get('volume_voxels', 0)) + 
                           ed.get('voxel_count', ed.get('volume_voxels', 0)) + 
                           et.get('voxel_count', et.get('volume_voxels', 0)))
        
        print(f"\n=== Calculated Total (NCR + ED + ET) ===")
        print(f"Should be: {calculated_mm3} mm³, {calculated_voxels} voxels")
    else:
        print("No results found")

if __name__ == "__main__":
    main()
