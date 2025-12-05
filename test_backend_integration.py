import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(endpoint):
    try:
        url = f"{BASE_URL}{endpoint}"
        print(f"\nTesting {url}...")
        response = requests.get(url)
        if response.status_code == 200:
            print("✅ Success!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ Failed with status {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Make sure the backend server is running on port 8000")
        return False

if __name__ == "__main__":
    print("="*50)
    print("Verifying Backend API & Model Status")
    print("="*50)
    
    # 1. Check Health
    if not test_endpoint("/health"):
        sys.exit(1)
        
    # 2. Check Model Status
    test_endpoint("/api/v1/models/status")
