-- Add missing columns to files table for metadata storage
-- This updates the existing schema to match the SQLAlchemy models

-- Add file_metadata column (renamed from 'metadata' to avoid reserved word)
ALTER TABLE files 
ADD COLUMN IF NOT EXISTS file_metadata JSONB;

-- Add patient_id column if it doesn't exist
ALTER TABLE files 
ADD COLUMN IF NOT EXISTS patient_id VARCHAR(255);

-- Verify the changes
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'files'
ORDER BY ordinal_position;
