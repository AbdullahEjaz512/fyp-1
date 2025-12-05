from backend.app.database import SessionLocal, CaseCollaboration, User
db = SessionLocal()

# Check all collaborations
collabs = db.query(CaseCollaboration).all()
print(f'Total collaborations: {len(collabs)}')
for c in collabs:
    primary = db.query(User).filter(User.user_id == c.primary_doctor_id).first()
    collab = db.query(User).filter(User.user_id == c.collaborating_doctor_id).first()
    primary_email = primary.email if primary else "Unknown"
    collab_email = collab.email if collab else "Unknown"
    print(f'  File {c.file_id}: {primary_email} shared with {collab_email} - Status: {c.status}')

# Check all users to see who can receive shared cases
print("\nAll doctors in system:")
users = db.query(User).filter(User.role != 'patient').all()
for u in users:
    print(f'  ID: {u.user_id}, Email: {u.email}, Name: {u.full_name}')

db.close()
