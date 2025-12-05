-- Add multi-doctor consultation support
-- Allows patients to grant access to multiple doctors for their files

-- 1. Create file_access_permissions table
CREATE TABLE IF NOT EXISTS file_access_permissions (
    permission_id SERIAL PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
    patient_id VARCHAR(100) NOT NULL,
    doctor_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    granted_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_level VARCHAR(50) DEFAULT 'view_and_analyze', -- 'view_only' or 'view_and_analyze'
    status VARCHAR(50) DEFAULT 'active', -- 'active', 'revoked'
    UNIQUE(file_id, doctor_id)
);

-- 2. Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_file_access_file ON file_access_permissions(file_id);
CREATE INDEX IF NOT EXISTS idx_file_access_doctor ON file_access_permissions(doctor_id);
CREATE INDEX IF NOT EXISTS idx_file_access_patient ON file_access_permissions(patient_id);

-- 3. Modify analysis_results to support multiple doctors analyzing same file
-- Already has file_id, so multiple records can exist for same file_id with different assessed_by values
-- Just ensure we have the assessed_by column
ALTER TABLE analysis_results 
ADD COLUMN IF NOT EXISTS assessed_by INTEGER REFERENCES users(user_id);

-- 4. Add doctor information to analysis results for easy retrieval
-- We'll join with users table to get doctor name, but let's cache it for performance
ALTER TABLE analysis_results
ADD COLUMN IF NOT EXISTS doctor_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS doctor_specialization VARCHAR(100);

-- 5. Comments for documentation
COMMENT ON TABLE file_access_permissions IS 'Controls which doctors can access and analyze specific patient files';
COMMENT ON COLUMN file_access_permissions.access_level IS 'Defines what the doctor can do: view_only or view_and_analyze';
COMMENT ON COLUMN file_access_permissions.status IS 'active or revoked - patient can revoke access';

-- 6. Create a view for easy querying of file access with doctor details
CREATE OR REPLACE VIEW file_access_with_doctor_info AS
SELECT 
    fap.permission_id,
    fap.file_id,
    fap.patient_id,
    fap.doctor_id,
    fap.granted_date,
    fap.access_level,
    fap.status,
    u.username as doctor_username,
    u.full_name as doctor_name,
    u.email as doctor_email,
    u.specialization as doctor_specialization,
    u.medical_license as doctor_license
FROM file_access_permissions fap
JOIN users u ON fap.doctor_id = u.user_id
WHERE fap.status = 'active';

-- 7. Create view for patient file history with all doctor analyses
CREATE OR REPLACE VIEW patient_file_analyses AS
SELECT 
    f.file_id,
    f.filename,
    f.patient_id,
    f.upload_date,
    f.status as file_status,
    ar.result_id,
    ar.assessed_by as doctor_id,
    ar.doctor_name,
    ar.doctor_specialization,
    ar.clinical_diagnosis,
    ar.doctor_interpretation,
    ar.assessment_date,
    ar.classification_type as tumor_classification,
    ar.classification_confidence as confidence_score
FROM files f
LEFT JOIN analysis_results ar ON f.file_id = ar.file_id
ORDER BY f.upload_date DESC, ar.assessment_date DESC;

COMMENT ON VIEW patient_file_analyses IS 'Shows all files with their corresponding doctor analyses';
