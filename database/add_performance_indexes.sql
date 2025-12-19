-- Performance Optimization: Additional Indexes for SegMind Database
-- Run this to improve query performance on frequently accessed columns

-- ============= Files Table Optimizations =============

-- Composite index for common queries: files by user and status
CREATE INDEX IF NOT EXISTS idx_files_user_status ON files(user_id, status);

-- Index for file searches by upload date (recent files first)
CREATE INDEX IF NOT EXISTS idx_files_created_at_desc ON files(created_at DESC);

-- Index for finding files by patient_id (multi-doctor access)
CREATE INDEX IF NOT EXISTS idx_files_patient_id ON files(patient_id);

-- ============= Analysis Results Optimizations =============

-- Composite index for finding latest analysis for a file
CREATE INDEX IF NOT EXISTS idx_analysis_file_created ON analysis_results(file_id, created_at DESC);

-- Index for filtering by tumor type in classification
CREATE INDEX IF NOT EXISTS idx_analysis_tumor_type ON analysis_results(classification_result);

-- Index for assessment queries
CREATE INDEX IF NOT EXISTS idx_analysis_assessed_by ON analysis_results(assessed_by);

-- ============= Access Control Optimizations =============

-- Composite index for doctor access checks (most frequent query)
CREATE INDEX IF NOT EXISTS idx_permissions_doctor_file ON file_access_permissions(doctor_id, file_id, status);

-- Composite index for patient file permissions
CREATE INDEX IF NOT EXISTS idx_permissions_patient_status ON file_access_permissions(patient_id, status);

-- Index for active permissions only
CREATE INDEX IF NOT EXISTS idx_permissions_status ON file_access_permissions(status) WHERE status = 'active';

-- ============= Collaboration Optimizations =============

-- Composite index for finding collaborations by primary doctor
CREATE INDEX IF NOT EXISTS idx_collab_primary_status ON case_collaborations(primary_doctor_id, status);

-- Composite index for finding collaborations for a specific doctor
CREATE INDEX IF NOT EXISTS idx_collab_collab_doctor_status ON case_collaborations(collaborating_doctor_id, status);

-- Index for recent collaborations
CREATE INDEX IF NOT EXISTS idx_collab_shared_at_desc ON case_collaborations(shared_at DESC);

-- ============= Discussion Optimizations =============

-- Composite index for finding discussions for a file
CREATE INDEX IF NOT EXISTS idx_discussion_file_created ON case_discussions(file_id, created_at DESC);

-- Index for finding all comments by a doctor
CREATE INDEX IF NOT EXISTS idx_discussion_doctor ON case_discussions(doctor_id);

-- ============= Audit Log Optimizations =============

-- Composite index for audit queries by user and time
CREATE INDEX IF NOT EXISTS idx_audit_user_created ON audit_logs(user_id, created_at DESC);

-- Index for filtering by action type
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_logs(action);

-- Composite index for resource access audits
CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_logs(resource_type, resource_id);

-- ============= Query Performance Analysis =============

-- Enable query statistics (run as superuser if needed)
-- This helps identify slow queries
ALTER DATABASE segmind_db SET log_min_duration_statement = 1000; -- Log queries > 1s

-- View table sizes and index usage
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS index_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- View index usage statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Identify unused indexes (candidates for removal)
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
    AND indexname NOT LIKE '%_pkey';

COMMIT;
