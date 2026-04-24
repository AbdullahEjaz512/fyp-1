
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app.database import get_db, File as DBFile, User, AnalysisResult

def check_db():
    try:
        db = next(get_db())
        print("Database connection successful.")
        
        users = db.query(User).all()
        print(f"Total Users: {len(users)}")
        for u in users:
            print(f" - User {u.user_id}: {u.username} ({u.role})")
            
        files = db.query(DBFile).all()
        print(f"\nTotal Files: {len(files)}")
        for f in files:
            print(f" - File {f.file_id}: {f.filename} (Patient: {f.patient_id}, Uploaded by: {f.user_id})")
            
        analyses = db.query(AnalysisResult).all()
        print(f"\nTotal Analyses: {len(analyses)}")
        for a in analyses:
            print(f" - Analysis {a.analysis_id} for File {a.file_id}: {a.classification_type}")
            
    except Exception as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    check_db()
