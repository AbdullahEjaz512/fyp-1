"""
Test the complete results flow:
1. Patient uploads file
2. Patient grants access to doctor  
3. Doctor analyzes file
4. Verify results appear correctly
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_complete_flow():
    print("\n" + "="*70)
    print("  TESTING COMPLETE RESULTS FLOW")
    print("="*70)
    
    # Step 1: Login as patient
    print("\n1. Login as Patient...")
    patient_login = {
        "email": "testupload@example.com",
        "password": "testpass123"
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/auth/login", json=patient_login)
    if response.status_code != 200:
        print(f"   ✗ Patient login failed: {response.status_code}")
        return
    
    patient_token = response.json()["access_token"]
    patient_id = response.json()["user"]["medical_record_number"]
    print(f"   ✓ Patient logged in")
    print(f"   Patient ID: {patient_id}")
    
    # Step 2: Get list of files
    print("\n2. Getting patient files...")
    headers = {"Authorization": f"Bearer {patient_token}"}
    response = requests.get(f"{BASE_URL}/api/v1/mri/files", headers=headers)
    
    if response.status_code != 200:
        print(f"   ✗ Failed to get files: {response.status_code}")
        return
    
    files = response.json()
    print(f"   ✓ Found {len(files)} files")
    
    if not files:
        print("   ⚠ No files found. Please upload a file first.")
        return
    
    # Get the first uploaded file
    test_file = None
    for f in files:
        if f["status"] in ["uploaded", "preprocessed"]:
            test_file = f
            break
    
    if not test_file:
        print("   ⚠ No uploaded files available for testing")
        test_file = files[0]  # Use any file
    
    file_id = test_file["file_id"]
    print(f"   Using file: {test_file['filename']} (ID: {file_id})")
    print(f"   Status: {test_file['status']}")
    
    # Step 3: Get list of doctors
    print("\n3. Getting list of doctors...")
    response = requests.get(f"{BASE_URL}/api/v1/doctors", headers=headers)
    
    if response.status_code != 200:
        print(f"   ✗ Failed to get doctors: {response.status_code}")
        return
    
    doctors_data = response.json()
    doctors = doctors_data.get("doctors", [])
    print(f"   ✓ Found {len(doctors)} doctors")
    
    if not doctors:
        print("   ⚠ No doctors available. Please create a doctor account.")
        return
    
    doctor = doctors[0]
    doctor_id = doctor["user_id"]
    print(f"   Using doctor: {doctor['full_name']} (ID: {doctor_id})")
    
    # Step 4: Grant access to doctor
    print(f"\n4. Granting access to doctor...")
    grant_data = {"doctor_ids": [doctor_id]}
    response = requests.post(
        f"{BASE_URL}/api/v1/files/{file_id}/grant-access",
        json=grant_data,
        headers=headers
    )
    
    if response.status_code == 200:
        print(f"   ✓ Access granted successfully")
    elif response.status_code == 400 and "already has access" in response.text:
        print(f"   ✓ Doctor already has access (that's fine)")
    else:
        print(f"   ✗ Failed to grant access: {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    # Step 5: Login as doctor
    print(f"\n5. Login as Doctor...")
    doctor_email = doctor.get("email")
    if not doctor_email:
        print("   ⚠ Doctor email not available, cannot test analysis")
        print("   Please create a doctor account with known credentials")
        return
    
    # For testing, we'll assume doctor password is "doctor123"
    # You may need to adjust this
    doctor_login = {
        "email": doctor_email,
        "password": "doctor123"  # Adjust if different
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/auth/login", json=doctor_login)
    if response.status_code != 200:
        print(f"   ⚠ Cannot login as doctor (password might be different)")
        print(f"   You'll need to manually test the analysis step")
        print(f"   Doctor email: {doctor_email}")
        return
    
    doctor_token = response.json()["access_token"]
    print(f"   ✓ Doctor logged in")
    
    # Step 6: Analyze the file
    print(f"\n6. Analyzing file...")
    doctor_headers = {"Authorization": f"Bearer {doctor_token}"}
    response = requests.post(
        f"{BASE_URL}/api/v1/analyze",
        params={"file_id": file_id},
        headers=doctor_headers
    )
    
    if response.status_code != 200:
        print(f"   ✗ Analysis failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    analysis_result = response.json()
    print(f"   ✓ Analysis completed successfully!")
    print(f"   Analysis ID: {analysis_result['analysis_id']}")
    print(f"   Diagnosis: {analysis_result['summary']['diagnosis']}")
    print(f"   Confidence: {analysis_result['summary']['confidence']}%")
    
    # Step 7: Get analysis results
    print(f"\n7. Retrieving analysis results...")
    time.sleep(1)  # Small delay to ensure database is updated
    
    response = requests.get(
        f"{BASE_URL}/api/v1/analyze/results/{file_id}",
        headers=doctor_headers
    )
    
    if response.status_code != 200:
        print(f"   ✗ Failed to get results: {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    results = response.json()
    print(f"   ✓ Results retrieved successfully!")
    print(f"   Total analyses: {results['total_analyses']}")
    print(f"   File: {results['filename']}")
    print(f"   Patient: {results['patient_id']}")
    
    # Step 8: Check doctor dashboard
    print(f"\n8. Checking doctor dashboard...")
    response = requests.get(
        f"{BASE_URL}/api/v1/doctors/dashboard",
        headers=doctor_headers
    )
    
    if response.status_code != 200:
        print(f"   ✗ Dashboard failed: {response.status_code}")
        print(f"   Response: {response.text}")
        return
    
    dashboard = response.json()
    print(f"   ✓ Dashboard loaded successfully!")
    print(f"   Assigned Patients: {dashboard['statistics']['assigned_patients']}")
    print(f"   Total Analyses: {dashboard['statistics']['total_analyses']}")
    print(f"   Recent Activities: {len(dashboard['recent_activities'])}")
    
    # Step 9: Check patient dashboard
    print(f"\n9. Checking patient dashboard (files list)...")
    response = requests.get(
        f"{BASE_URL}/api/v1/mri/files",
        headers=headers
    )
    
    if response.status_code != 200:
        print(f"   ✗ Failed to get files: {response.status_code}")
        return
    
    files = response.json()
    analyzed_count = sum(1 for f in files if f["status"] == "analyzed")
    print(f"   ✓ Patient has {len(files)} files")
    print(f"   Analyzed: {analyzed_count}")
    
    print("\n" + "="*70)
    print("✓ COMPLETE FLOW TEST PASSED!")
    print("="*70)
    print("\nNext steps:")
    print("1. Open frontend at http://localhost:5174")
    print("2. Login as patient and verify file shows 'View Results' button")
    print("3. Login as doctor and verify dashboard shows data")
    print("4. Click 'Analyze' on a new file and verify it navigates to results")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        test_complete_flow()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
