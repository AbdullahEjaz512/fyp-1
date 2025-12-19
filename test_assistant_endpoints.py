"""
Quick test script for Assistant API endpoints
Run with: python test_assistant_endpoints.py
"""
import requests
import json
import base64

BASE_URL = "http://localhost:8000"

print("\n" + "="*70)
print("  TESTING ASSISTANT API ENDPOINTS")
print("="*70)

# Step 1: Get test token
print("\n1. Loading test token...")
try:
    with open("test_token.txt", "r") as f:
        token = f.read().strip()
    headers = {"Authorization": f"Bearer {token}"}
    print(f"   ✓ Token loaded")
except Exception as e:
    print(f"   ✗ Error loading token: {e}")
    print("   Run: python create_test_user.py first")
    exit(1)

# Step 2: Test chat endpoint
print("\n2. Testing chat endpoint...")
try:
    response = requests.post(
        f"{BASE_URL}/api/v1/assistant/chat",
        headers=headers,
        json={"message": "What is validation Dice?"}
    )
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Chat successful!")
        print(f"   Answer: {result['answer'][:100]}...")
        if result.get('sources'):
            print(f"   Sources: {len(result['sources'])} docs found")
    else:
        print(f"   ✗ Chat failed: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Step 3: Test report generation
print("\n3. Testing text report generation...")
report_data = {
    "patient_id": "PT-2025-00001",
    "doctor_name": "Dr. Test",
    "summary": "Adult patient with suspected glioma. MRI T1/T2/FLAIR acquired.",
    "classification": {"type": "Glioma", "confidence": "0.87"},
    "segmentation": {"volume": "28.3 cc", "dice": "0.81"},
    "notes": "Recommend biopsy and correlate with histopathology."
}

try:
    response = requests.post(
        f"{BASE_URL}/api/v1/assistant/report",
        headers=headers,
        json=report_data
    )
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ Report generated!")
        print(f"   Preview:\n{result['report_text'][:200]}...")
    else:
        print(f"   ✗ Report failed: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Step 4: Test PDF report generation
print("\n4. Testing PDF report generation...")
try:
    response = requests.post(
        f"{BASE_URL}/api/v1/assistant/report/pdf",
        headers=headers,
        json=report_data
    )
    if response.status_code == 200:
        result = response.json()
        pdf_b64 = result.get('pdf_base64', '')
        if pdf_b64:
            # Optionally save to file
            pdf_bytes = base64.b64decode(pdf_b64)
            with open("test_report.pdf", "wb") as f:
                f.write(pdf_bytes)
            print(f"   ✓ PDF generated! ({len(pdf_bytes)} bytes)")
            print(f"   Saved to: test_report.pdf")
        else:
            print(f"   ✗ No PDF data in response")
    else:
        print(f"   ✗ PDF failed: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Step 5: Test similar cases (if any exist)
print("\n5. Testing similar cases endpoint...")
try:
    response = requests.get(
        f"{BASE_URL}/api/v1/assistant/cases/1/similar",
        headers=headers
    )
    if response.status_code == 200:
        result = response.json()
        similar = result.get('similar', [])
        print(f"   ✓ Similar cases found: {len(similar)}")
        if similar:
            print(f"   First case: {similar[0]}")
    else:
        print(f"   ✗ Similar cases failed: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*70)
print("✓ Assistant API testing complete!")
print("="*70 + "\n")
