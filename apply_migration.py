"""
Apply database migration: Add notes column to files table
"""
import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'backend', 'brain_tumor.db')

print("============================================================")
print("  Applying Database Migration for Discussion Panel (FE-5)  ")
print("============================================================\n")

if not os.path.exists(db_path):
    print(f"✗ Database not found at {db_path}")
    exit(1)

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if column exists
    cursor.execute("PRAGMA table_info(files)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'notes' not in columns:
        print("Adding 'notes' column to files table...")
        cursor.execute("ALTER TABLE files ADD COLUMN notes TEXT")
        conn.commit()
        print("✓ Successfully added 'notes' column to files table")
    else:
        print("✓ 'notes' column already exists in files table")
    
    conn.close()
    print("\n✓ Database migration completed successfully!\n")
    print("Now restart the backend server:")
    print("  cd backend")
    print("  python app/main.py")
    print("\nThen run the test:")
    print("  python test_module1_features.py")
    
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)
