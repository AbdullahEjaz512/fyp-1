"""
Create a test user for upload testing
"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("\n" + "="*70)
print("  CREATE TEST USER FOR UPLOAD TESTING")
print("="*70)

# Create a new test user
test_user = {
    "email": "testupload@example.com",
    "username": "testupload",
    "password": "testpass123",
    "role": "patient",
    "full_name": "Test Upload User"
}

print("\n1. Creating test user...")
print(f"   Email: {test_user['email']}")
print(f"   Password: {test_user['password']}")

try:
    response = requests.post(f"{BASE_URL}/api/v1/auth/register", json=test_user)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ User created successfully!")
        print(f"   User ID: {result.get('user_id')}")
    elif response.status_code == 400 and "already exists" in response.text.lower():
        print(f"   ✓ User already exists (that's fine)")
    else:
        print(f"   ✗ Registration failed: {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Test login
print("\n2. Testing login...")

try:
    login_data = {
        "email": test_user["email"],
        "password": test_user["password"]
    }
    
    response = requests.post(f"{BASE_URL}/api/v1/auth/login", json=login_data)
    
    if response.status_code == 200:
        result = response.json()
        token = result.get("access_token")
        print(f"   ✓ Login successful!")
        print(f"   Token: {token[:30]}...")
        
        # Save token to file for upload test
        with open("test_token.txt", "w") as f:
            f.write(token)
        print(f"   Token saved to test_token.txt")
        
    else:
        print(f"   ✗ Login failed: {response.status_code}")
        print(f"   Response: {response.text}")
        
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*70)
print("✓ Test user ready! You can now test file upload.")
print("="*70 + "\n")
