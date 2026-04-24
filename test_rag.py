
import os
import sys
import time
from fastapi.testclient import TestClient
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app.main import app
from backend.app.routers.assistant import get_current_user as assistant_get_current_user
from backend.app.database import get_db

def test_rag_working():
    # Override dependencies
    app.dependency_overrides[assistant_get_current_user] = lambda: {"user_id": 24, "role": "doctor", "username": "dr_test"}
    app.dependency_overrides[get_db] = lambda: next(get_db())

    client = TestClient(app)
    headers = {"Authorization": "Bearer faketoken"}
    
    print("\n=======================================================")
    print("--- Comprehensive LLM Assistant Testing (Groq API) ---")
    print("=======================================================\n")
    
    questions = [
        {
            "category": "Medical Knowledge",
            "message": "Tell me about Anaplastic Astrocytoma, including its WHO grade and typical prognosis."
        },
        {
            "category": "Platform Feature",
            "message": "How do I upload a scan and what happens after I upload it?"
        },
        {
            "category": "Tumor Subregions",
            "message": "What is the difference between the NCR and ET regions in a brain tumor?"
        },
        {
            "category": "Patient Query (Simulated)",
            "message": "What does file_id 51 show?" 
        }
    ]

    for idx, q in enumerate(questions):
        print(f"[{idx+1}] Category: {q['category']}")
        print(f"    Question: \"{q['message']}\"")
        
        start_time = time.time()
        response = client.post("/api/v1/assistant/chat", json={"message": q['message']}, headers=headers)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            raw_response = data.get('response', '')
            clean_response = raw_response.encode('ascii', 'ignore').decode('ascii')
            
            print(f"    Status: SUCCESS (200 OK)")
            print(f"    Response Time: {elapsed_time:.2f} seconds")
            
            if "Anaplastic" in q['message']:
                print(f"    Full Response:\n{clean_response}\n")
            else:
                print(f"    Response Snippet: {clean_response[:200]}...\n")
            
        else:
            print(f"    Status: FAILED ({response.status_code})")
            print(f"    Detail: {response.json()}\n")

if __name__ == "__main__":
    test_rag_working()
