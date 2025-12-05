import requests
import sys

BASE_URL = "http://127.0.0.1:8000/api/v1"

def list_users():
    # We can't list users without being admin, but we can try to login with known users
    # Or we can check the database directly if we had a script for that.
    # Since I can't easily run SQL, I'll try to register a new user and see if it works.
    
    email = "debug_user@example.com"
    password = "password123"
    
    print(f"Attempting to register {email}...")
    register_data = {
        "email": email,
        "password": password,
        "role": "doctor",
        "full_name": "Debug User",
        "medical_license": "DBG-001"
    }
    
    try:
        resp = requests.post(f"{BASE_URL}/auth/register", json=register_data)
        if resp.status_code == 200:
            print("✅ Registration successful")
        elif resp.status_code == 400 and "already exists" in resp.text:
            print("ℹ️ User already exists")
        else:
            print(f"❌ Registration failed: {resp.status_code} - {resp.text}")
            
        print(f"Attempting to login as {email}...")
        login_data = {
            "email": email,
            "password": password
        }
        
        resp = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        if resp.status_code == 200:
            print("✅ Login successful!")
            print(f"Token: {resp.json().get('access_token')[:20]}...")
            return True
        else:
            print(f"❌ Login failed: {resp.status_code} - {resp.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

if __name__ == "__main__":
    list_users()
