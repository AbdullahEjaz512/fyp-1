-- Add notes column to files table for discussion panel
ALTER TABLE files ADD COLUMN IF NOT EXISTS notes TEXT;

-- Add comment
COMMENT ON COLUMN files.notes IS 'Stores discussion/comments as JSON for multi-doctor collaboration (FE-5)';
