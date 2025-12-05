# Database Configuration for Seg-Mind
# PostgreSQL Connection Settings

DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "segmind_db",
    "user": "postgres",
    "password": "postgres123"
}

# SQLAlchemy Database URL
DATABASE_URL = f"postgresql://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"

# Alternative async URL for async operations
ASYNC_DATABASE_URL = f"postgresql+asyncpg://{DATABASE_CONFIG['user']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
