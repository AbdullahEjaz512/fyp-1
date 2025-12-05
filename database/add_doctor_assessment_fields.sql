-- Add doctor assessment fields to analysis_results table
-- Run this to enable doctor's clinical interpretation and prescription

ALTER TABLE analysis_results 
ADD COLUMN IF NOT EXISTS doctor_interpretation TEXT,
ADD COLUMN IF NOT EXISTS clinical_diagnosis VARCHAR(500),
ADD COLUMN IF NOT EXISTS prescription TEXT,
ADD COLUMN IF NOT EXISTS treatment_plan TEXT,
ADD COLUMN IF NOT EXISTS follow_up_notes TEXT,
ADD COLUMN IF NOT EXISTS next_appointment DATE,
ADD COLUMN IF NOT EXISTS assessment_date TIMESTAMP,
ADD COLUMN IF NOT EXISTS assessed_by INTEGER REFERENCES users(user_id);

-- Add index for faster queries
CREATE INDEX IF NOT EXISTS idx_analysis_assessed_by ON analysis_results(assessed_by);

-- Add comment
COMMENT ON COLUMN analysis_results.doctor_interpretation IS 'Doctor''s professional interpretation of the AI analysis results';
COMMENT ON COLUMN analysis_results.clinical_diagnosis IS 'Final clinical diagnosis by the doctor';
COMMENT ON COLUMN analysis_results.prescription IS 'Medications and dosages prescribed';
COMMENT ON COLUMN analysis_results.treatment_plan IS 'Recommended treatment approach';
COMMENT ON COLUMN analysis_results.follow_up_notes IS 'Follow-up instructions and monitoring plan';
COMMENT ON COLUMN analysis_results.next_appointment IS 'Scheduled next appointment date';
COMMENT ON COLUMN analysis_results.assessment_date IS 'When doctor completed the assessment';
COMMENT ON COLUMN analysis_results.assessed_by IS 'User ID of the doctor who provided the assessment';
