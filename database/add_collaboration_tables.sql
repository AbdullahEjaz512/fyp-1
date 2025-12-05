-- Multi-Doctor Collaboration Tables
-- Run this migration to add collaboration support

-- Case Collaboration table - tracks which doctors are collaborating on a case
CREATE TABLE IF NOT EXISTS case_collaborations (
    collaboration_id SERIAL PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
    primary_doctor_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    collaborating_doctor_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    shared_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active',
    message TEXT,
    
    -- Prevent duplicate collaborations
    UNIQUE(file_id, collaborating_doctor_id)
);

-- Case Discussion table - comments/discussion thread for a case
CREATE TABLE IF NOT EXISTS case_discussions (
    discussion_id SERIAL PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
    doctor_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    comment TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    parent_id INTEGER REFERENCES case_discussions(discussion_id) ON DELETE CASCADE
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_collaborations_file ON case_collaborations(file_id);
CREATE INDEX IF NOT EXISTS idx_collaborations_primary_doctor ON case_collaborations(primary_doctor_id);
CREATE INDEX IF NOT EXISTS idx_collaborations_collaborating_doctor ON case_collaborations(collaborating_doctor_id);
CREATE INDEX IF NOT EXISTS idx_collaborations_status ON case_collaborations(status);

CREATE INDEX IF NOT EXISTS idx_discussions_file ON case_discussions(file_id);
CREATE INDEX IF NOT EXISTS idx_discussions_doctor ON case_discussions(doctor_id);
CREATE INDEX IF NOT EXISTS idx_discussions_created ON case_discussions(created_at);

-- Grant permissions (adjust if using different user)
-- GRANT ALL ON case_collaborations TO segmind_user;
-- GRANT ALL ON case_discussions TO segmind_user;
-- GRANT USAGE, SELECT ON SEQUENCE case_collaborations_collaboration_id_seq TO segmind_user;
-- GRANT USAGE, SELECT ON SEQUENCE case_discussions_discussion_id_seq TO segmind_user;
