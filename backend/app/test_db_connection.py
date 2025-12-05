"""
Test PostgreSQL Database Connection
"""
import psycopg2
from database_config import DATABASE_CONFIG

def test_connection():
    """Test database connection"""
    try:
        # Connect to database
        conn = psycopg2.connect(**DATABASE_CONFIG)
        cursor = conn.cursor()
        
        # Test query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print("✓ Database connection successful!")
        print(f"PostgreSQL version: {version[0]}")
        
        # Check tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        print(f"\n✓ Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Check admin user
        cursor.execute("SELECT COUNT(*) FROM users;")
        user_count = cursor.fetchone()[0]
        print(f"\n✓ Users in database: {user_count}")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()
