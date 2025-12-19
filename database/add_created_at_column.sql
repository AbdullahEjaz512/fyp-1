-- Add created_at column to files table for consistency with indexes
-- This allows better performance tracking and aligns with index definitions

ALTER TABLE files 
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Update created_at to match upload_date for existing records
UPDATE files 
SET created_at = upload_date 
WHERE created_at IS NULL;

-- Verify the changes
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'files' AND column_name IN ('created_at', 'upload_date')
ORDER BY ordinal_position;
