-- Add default_branch column to projects table
-- This stores the repository's default branch (main, master, etc.)

ALTER TABLE projects
ADD COLUMN IF NOT EXISTS default_branch VARCHAR(255) DEFAULT 'main';

-- Update existing projects to have 'main' as default
UPDATE projects
SET default_branch = 'main'
WHERE default_branch IS NULL;

-- Add comment
COMMENT ON COLUMN projects.default_branch IS 'Default branch of the GitHub repository (e.g., main, master)';

