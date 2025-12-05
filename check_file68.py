from backend.app.database import SessionLocal, File, User, CaseCollaboration
db = SessionLocal()

# Check file 68 details
f = db.query(File).filter(File.file_id == 68).first()
if f:
    uploader = db.query(User).filter(User.user_id == f.user_id).first()
    print("File 68:")
    print(f"  Filename: {f.filename}")
    print(f"  Uploaded by user_id: {f.user_id} ({uploader.email if uploader else 'Unknown'})")
    print(f"  Patient ID: {f.patient_id}")

# Check collaborations for file 68
collabs = db.query(CaseCollaboration).filter(CaseCollaboration.file_id == 68).all()
print(f"\nCollaborations for file 68: {len(collabs)}")
for c in collabs:
    primary = db.query(User).filter(User.user_id == c.primary_doctor_id).first()
    collab_doc = db.query(User).filter(User.user_id == c.collaborating_doctor_id).first()
    print(f"  Primary: {primary.email if primary else 'Unknown'} (ID: {c.primary_doctor_id})")
    print(f"  Collaborator: {collab_doc.email if collab_doc else 'Unknown'} (ID: {c.collaborating_doctor_id})")

db.close()
