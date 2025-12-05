"""
Apply database migration: Add notes column to files table (PostgreSQL)
"""
import psycopg2
from psycopg2 import sql

# Database connection details
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "segmind_db",
    "user": "postgres",
    "password": "postgres123"
}

print("============================================================")
print("  Applying Database Migration for Discussion Panel (FE-5)  ")
print("============================================================\n")

try:
    # Connect to PostgreSQL
    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("✓ Connected to database")
    
    # Check if column exists
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='files' AND column_name='notes'
    """)
    
    if cursor.fetchone() is None:
        print("\nAdding 'notes' column to files table...")
        cursor.execute("ALTER TABLE files ADD COLUMN notes TEXT")
        conn.commit()
        print("✓ Successfully added 'notes' column to files table")
    else:
        print("✓ 'notes' column already exists in files table")
    
    cursor.close()
    conn.close()
    
    print("\n✓ Database migration completed successfully!\n")
    print("Now restart the backend server:")
    print("  1. Stop the current backend server (Ctrl+C)")
    print("  2. cd backend")
    print("  3. python app/main.py")
    print("\nThen run the test:")
    print("  python test_module1_features.py")
    
except psycopg2.OperationalError as e:
    print(f"✗ Cannot connect to database: {e}")
    print("\nMake sure PostgreSQL is running:")
    print("  1. Check if postgres service is running")
    print("  2. Verify credentials in database_config.py")
    exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)
