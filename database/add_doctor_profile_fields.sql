-- Add doctor profile fields to users table
-- Module 6: Doctor Profile Management

-- Add new columns for doctor profiles
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS full_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS medical_license VARCHAR(100),
ADD COLUMN IF NOT EXISTS specialization VARCHAR(100),
ADD COLUMN IF NOT EXISTS institution VARCHAR(255),
ADD COLUMN IF NOT EXISTS department VARCHAR(100),
ADD COLUMN IF NOT EXISTS years_of_experience INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS phone_number VARCHAR(50),
ADD COLUMN IF NOT EXISTS profile_picture_url TEXT,
ADD COLUMN IF NOT EXISTS bio TEXT;

-- Add patient profile fields
ALTER TABLE users
ADD COLUMN IF NOT EXISTS date_of_birth DATE,
ADD COLUMN IF NOT EXISTS gender VARCHAR(20),
ADD COLUMN IF NOT EXISTS medical_record_number VARCHAR(100);

-- Add indexes for commonly queried fields
CREATE INDEX IF NOT EXISTS idx_users_medical_license ON users(medical_license);
CREATE INDEX IF NOT EXISTS idx_users_specialization ON users(specialization);
CREATE INDEX IF NOT EXISTS idx_users_institution ON users(institution);

-- Update existing users to have full_name from username
UPDATE users SET full_name = username WHERE full_name IS NULL;

COMMIT;
