"""
Apply database migration to add created_at column to files table
"""
import psycopg2
from psycopg2 import sql

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "segmind_db",
    "user": "postgres",
    "password": "postgres123"
}

def apply_migration():
    """Apply the created_at column migration"""
    print("🔄 Applying database migration...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Add created_at column
        print("  Adding created_at column...")
        cursor.execute("""
            ALTER TABLE files 
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
        """)
        
        # Update existing records
        print("  Updating existing records...")
        cursor.execute("""
            UPDATE files 
            SET created_at = upload_date 
            WHERE created_at IS NULL;
        """)
        
        # Verify
        cursor.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'files' AND column_name IN ('created_at', 'upload_date')
            ORDER BY ordinal_position;
        """)
        
        results = cursor.fetchall()
        print("\n✅ Migration applied successfully!")
        print("\nVerified columns:")
        for row in results:
            print(f"  - {row[0]}: {row[1]} (nullable: {row[2]})")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    apply_migration()
