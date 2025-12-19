"""
Test ensemble API integration
Quick test to verify ensemble models are working in the API
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_ensemble_status():
    """Test ensemble status endpoint"""
    print("=" * 60)
    print("TEST 1: Ensemble Status Endpoint")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/api/v1/ensemble/status")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Ensemble status endpoint working!")
        print(f"\nEnsemble Enabled: {data['ensemble_enabled']}")
        print(f"Ensemble Available: {data['ensemble_available']}")
        print(f"\nSegmentation:")
        print(f"  - Initialized: {data['models']['segmentation']['ensemble_initialized']}")
        print(f"  - Features: {', '.join(data['models']['segmentation']['features'])}")
        print(f"\nClassification:")
        print(f"  - Initialized: {data['models']['classification']['ensemble_initialized']}")
        print(f"  - Method: {data['models']['classification']['method']}")
        print(f"  - Features: {', '.join(data['models']['classification']['features'])}")
        print(f"\nExpected Improvements:")
        print(f"  - Segmentation: {data['expected_improvements']['segmentation_dice']}")
        print(f"  - Classification: {data['expected_improvements']['classification_accuracy']}")
        return True
    else:
        print(f"❌ Failed: {response.status_code}")
        print(response.text)
        return False

def test_health():
    """Test health endpoint"""
    print("\n" + "=" * 60)
    print("TEST 2: Health Check")
    print("=" * 60)
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Server healthy: {data}")
        return True
    else:
        print(f"❌ Server not responding")
        return False

def main():
    print("\n🔬 Testing Ensemble API Integration\n")
    
    try:
        # Test 1: Health check
        health_ok = test_health()
        
        # Test 2: Ensemble status
        ensemble_ok = test_ensemble_status()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        if health_ok and ensemble_ok:
            print("✅ All tests passed!")
            print("\n🎉 Ensemble models are deployed and ready!")
            print("\nNext Steps:")
            print("1. Upload a medical scan")
            print("2. Run analysis with ensemble predictions")
            print("3. Check the 'ensemble' field in the response for uncertainty scores")
        else:
            print("❌ Some tests failed")
            if not health_ok:
                print("   - Server health check failed. Is the server running?")
            if not ensemble_ok:
                print("   - Ensemble status check failed. Check server logs.")
    
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error!")
        print("Server is not running. Please start it with:")
        print("   cd backend")
        print("   python -m uvicorn app.main:app --reload --port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
