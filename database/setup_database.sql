-- Seg-Mind Database Setup Script
-- PostgreSQL Database for Brain Tumor Analysis System

-- Create database (run this as postgres user)
-- CREATE DATABASE segmind_db;

-- Connect to the database
\c segmind_db;

-- Create Users table
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'doctor', 'radiologist', 'oncologist', 'patient')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Files table
CREATE TABLE IF NOT EXISTS files (
    file_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    safe_filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL CHECK (file_type IN ('DICOM', 'NIfTI')),
    file_path TEXT NOT NULL,
    file_size BIGINT,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preprocessed BOOLEAN DEFAULT FALSE,
    preprocessed_path TEXT,
    preprocessing_params JSONB,
    metadata JSONB,
    status VARCHAR(50) DEFAULT 'uploaded' CHECK (status IN ('uploaded', 'preprocessing', 'preprocessed', 'analyzing', 'analyzed', 'failed'))
);

-- Create Analysis Results table
CREATE TABLE IF NOT EXISTS analysis_results (
    result_id SERIAL PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Segmentation results
    segmentation_data JSONB,
    segmentation_confidence FLOAT,
    tumor_volume FLOAT,
    tumor_regions JSONB,
    
    -- Classification results
    classification_type VARCHAR(100),
    classification_confidence FLOAT,
    who_grade VARCHAR(20),
    malignancy_level VARCHAR(50),
    
    -- Performance metrics
    preprocessing_time FLOAT,
    segmentation_time FLOAT,
    classification_time FLOAT,
    total_time FLOAT,
    
    -- Additional metadata
    model_versions JSONB,
    notes TEXT
);

-- Create indexes for better query performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_files_user_id ON files(user_id);
CREATE INDEX idx_files_status ON files(status);
CREATE INDEX idx_files_upload_date ON files(upload_date);
CREATE INDEX idx_analysis_file_id ON analysis_results(file_id);
CREATE INDEX idx_analysis_date ON analysis_results(analysis_date);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for users table
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default admin user (password: admin123 - hashed with bcrypt)
-- Note: This is a placeholder. In production, hash properly using bcrypt
INSERT INTO users (email, username, password_hash, role) 
VALUES (
    'admin@segmind.com', 
    'admin', 
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYIBx8fK3qC', -- password: admin123
    'admin'
) ON CONFLICT (email) DO NOTHING;

-- Create a view for file statistics
CREATE OR REPLACE VIEW file_statistics AS
SELECT 
    u.user_id,
    u.username,
    u.role,
    COUNT(f.file_id) as total_files,
    COUNT(CASE WHEN f.preprocessed = true THEN 1 END) as preprocessed_files,
    COUNT(CASE WHEN f.status = 'analyzed' THEN 1 END) as analyzed_files,
    SUM(f.file_size) as total_storage_bytes
FROM users u
LEFT JOIN files f ON u.user_id = f.user_id
GROUP BY u.user_id, u.username, u.role;

-- Grant permissions (adjust as needed for your user)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO your_app_user;

-- Display table structure
\dt
\d users
\d files
\d analysis_results

-- Show success message
SELECT 'Database setup completed successfully!' as status;
