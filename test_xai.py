
import os
import sys
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
# Hack to make imports match exactly what FastAPI uses
import app.dependencies.auth as auth_deps
from backend.app.main import app
from backend.app.database import get_db

def test_xai_local():
    app.dependency_overrides[auth_deps.get_current_user] = lambda: {"user_id": 24, "role": "doctor", "username": "testdoctor"}
    app.dependency_overrides[get_db] = lambda: next(get_db())

    client = TestClient(app)
    headers = {"Authorization": "Bearer faketoken"}
    payload = {
        "file_id": 51,
        "method": "gradcam"
    }
    
    response = client.post("/api/v1/advanced/explain/classification", json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        note = data.get("note", "No note")
        print("Status: 200 OK")
        print(f"Target Class: {data.get('target_class')}")
        print(f"Confidence: {data.get('confidence')}")
        print(f"Note: {note}")
        
        b64 = data.get("heatmap_base64", "")
        if b64.startswith("iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAIAAADTED8xAAAHIklEQVR4"):
            print("Heatmap: DUMMY DETECTED")
        else:
            print("Heatmap: REAL HEATMAP GENERATED")
            print(f"Base64 length: {len(b64)}")
    else:
        print(f"Failed with status: {response.status_code}")
        print(response.json())

if __name__ == "__main__":
    test_xai_local()
