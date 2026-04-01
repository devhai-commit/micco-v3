-- ═══════════════════════════════════════════════════════════════════════════
-- MiccoRAG-v3 — Full schema (tổng hợp từ tất cả migrations)
-- Thứ tự migration: init → 002_rls → 003_document_versions → 004_document_fields
-- Safe to re-run: dùng IF NOT EXISTS / ON CONFLICT DO NOTHING
-- ═══════════════════════════════════════════════════════════════════════════

-- ─── Extensions ───────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";     -- pgvector cho embeddings

-- ═══════════════════════════════════════════════════════════════════════════
-- TABLES (theo thứ tự dependency)
-- ═══════════════════════════════════════════════════════════════════════════

-- ─── Departments ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS departments (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_departments_name ON departments (name);

-- ─── Users ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(100) NOT NULL,
    email           VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    role            VARCHAR(50)  NOT NULL DEFAULT 'Nhân viên',
    department_id   INTEGER      REFERENCES departments(id) ON DELETE SET NULL,
    avatar          VARCHAR(500),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email      ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_role       ON users (role);
CREATE INDEX IF NOT EXISTS idx_users_department ON users (department_id);

-- ─── Knowledge Bases (Workspaces) ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id                 SERIAL PRIMARY KEY,
    name               VARCHAR(255)  NOT NULL,
    description        TEXT,
    system_prompt      TEXT,
    kg_language        VARCHAR(50),
    kg_entity_types    JSONB,
    search_mode        VARCHAR(50)   DEFAULT 'hybrid',
    suggested_questions JSONB,
    created_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

-- ─── Knowledge Entries ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS knowledge_entries (
    id              SERIAL PRIMARY KEY,
    title           VARCHAR(500) NOT NULL,
    content_html    TEXT         NOT NULL DEFAULT '',
    content_text    TEXT         NOT NULL DEFAULT '',
    category        VARCHAR(100) NOT NULL DEFAULT 'Chung',
    tags            JSONB        NOT NULL DEFAULT '[]'::jsonb,
    owner_id        INTEGER      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    department_id   INTEGER      REFERENCES departments(id) ON DELETE SET NULL,
    visibility      VARCHAR(20)  NOT NULL DEFAULT 'internal',
    status          VARCHAR(20)  NOT NULL DEFAULT 'Active',
    ingest_status   VARCHAR(20)           DEFAULT 'pending',
    ingest_error    TEXT,
    approval_status VARCHAR(20)  NOT NULL DEFAULT 'pending_approval',
    approval_note   TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_owner      ON knowledge_entries (owner_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_status     ON knowledge_entries (status);
CREATE INDEX IF NOT EXISTS idx_knowledge_department ON knowledge_entries (department_id);

-- ─── Documents ────────────────────────────────────────────────────────────
-- Ghi chú: enum DocumentStatus = pending|parsing|processing|indexing|indexed|failed
CREATE TABLE IF NOT EXISTS documents (
    id                  SERIAL PRIMARY KEY,
    workspace_id        INTEGER      NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    filename            VARCHAR(255) NOT NULL,
    original_filename   VARCHAR(255) NOT NULL,
    file_type           VARCHAR(50)  NOT NULL,
    file_size           INTEGER      NOT NULL,
    status              VARCHAR(20)  NOT NULL DEFAULT 'pending',  -- DocumentStatus enum
    chunk_count         INTEGER      NOT NULL DEFAULT 0,
    error_message       VARCHAR(500),
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- Approvals & Access Control
    uploader_id         INTEGER      REFERENCES users(id) ON DELETE SET NULL,
    department_id       INTEGER      REFERENCES departments(id) ON DELETE SET NULL,
    visibility          VARCHAR(20)  NOT NULL DEFAULT 'internal',
    approval_status     VARCHAR(20)  NOT NULL DEFAULT 'pending',
    approval_note       TEXT,

    -- NexusRAG parsing metadata
    markdown_content    TEXT,
    page_count          INTEGER      NOT NULL DEFAULT 0,
    image_count         INTEGER      NOT NULL DEFAULT 0,
    table_count         INTEGER      NOT NULL DEFAULT 0,
    parser_version      VARCHAR(50),               -- "docling" | "legacy"
    processing_time_ms  INTEGER      NOT NULL DEFAULT 0,

    -- migration 004: document metadata
    category            VARCHAR(100),
    tags                TEXT,                      -- comma-separated
    thumbnail           VARCHAR(255)               -- filename of thumbnail
);

CREATE INDEX IF NOT EXISTS idx_documents_workspace   ON documents (workspace_id);
CREATE INDEX IF NOT EXISTS idx_documents_uploader    ON documents (uploader_id);
CREATE INDEX IF NOT EXISTS idx_documents_department  ON documents (department_id);
CREATE INDEX IF NOT EXISTS idx_documents_status      ON documents (status);

-- ─── Document Images ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS document_images (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    image_id    VARCHAR(100) NOT NULL UNIQUE,   -- UUID
    page_no     INTEGER      NOT NULL DEFAULT 0,
    file_path   VARCHAR(500) NOT NULL,
    caption     TEXT         NOT NULL DEFAULT '',
    width       INTEGER      NOT NULL DEFAULT 0,
    height      INTEGER      NOT NULL DEFAULT 0,
    mime_type   VARCHAR(50)  NOT NULL DEFAULT 'image/png',
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doc_images_document ON document_images (document_id);

-- ─── Document Tables ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS document_tables (
    id               SERIAL PRIMARY KEY,
    document_id      INTEGER      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    table_id         VARCHAR(100) NOT NULL UNIQUE,
    page_no          INTEGER      NOT NULL DEFAULT 0,
    content_markdown TEXT         NOT NULL DEFAULT '',
    caption          TEXT         NOT NULL DEFAULT '',
    num_rows         INTEGER      NOT NULL DEFAULT 0,
    num_cols         INTEGER      NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doc_tables_document ON document_tables (document_id);

-- ─── Document Versions (migration 003) ────────────────────────────────────
CREATE TABLE IF NOT EXISTS document_versions (
    id               SERIAL PRIMARY KEY,
    document_id      INTEGER      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version_number   INTEGER      NOT NULL,
    version_label    VARCHAR(50)  NOT NULL DEFAULT 'V 1.0',
    filename         VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255),
    file_size        INTEGER               DEFAULT 0,
    change_note      TEXT,
    created_by       INTEGER      REFERENCES users(id) ON DELETE CASCADE,
    is_current       BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMPTZ           DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_document_versions_id          ON document_versions (id);
CREATE INDEX IF NOT EXISTS ix_document_versions_document_id ON document_versions (document_id);

-- ─── Chat Messages ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_messages (
    id               SERIAL PRIMARY KEY,
    workspace_id     INTEGER      NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    user_id          INTEGER      REFERENCES users(id) ON DELETE SET NULL,
    message_id       VARCHAR(50)  NOT NULL,
    role             VARCHAR(20)  NOT NULL,
    content          TEXT         NOT NULL,
    sources          JSONB,
    related_entities JSONB,
    image_refs       JSONB,
    thinking         TEXT,
    ratings          JSONB,
    agent_steps      JSONB,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_workspace  ON chat_messages (workspace_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_user       ON chat_messages (user_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_message_id ON chat_messages (message_id);

-- ─── System Chat Logs ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS system_chat_logs (
    id            SERIAL PRIMARY KEY,
    workspace_id  INTEGER      NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    ip_address    VARCHAR(50),
    timestamp     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    response_time FLOAT        NOT NULL,
    question      TEXT         NOT NULL,
    answer        TEXT         NOT NULL,
    method        VARCHAR(50)  NOT NULL   -- hybrid, vector_only, graph_only, ...
);

CREATE INDEX IF NOT EXISTS idx_system_chat_logs_workspace ON system_chat_logs (workspace_id);

-- ═══════════════════════════════════════════════════════════════════════════
-- ROW-LEVEL SECURITY (migration 002)
-- ═══════════════════════════════════════════════════════════════════════════

-- ─── documents RLS ────────────────────────────────────────────────────────
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS doc_admin_policy ON documents;
CREATE POLICY doc_admin_policy ON documents
    FOR ALL USING (current_setting('app.user_role', true) = 'Admin');

DROP POLICY IF EXISTS doc_dept_policy ON documents;
CREATE POLICY doc_dept_policy ON documents
    FOR ALL USING (
        visibility = 'public'
        OR department_id = (current_setting('app.user_dept_id', true)::int)
        OR uploader_id   = (current_setting('app.user_id',      true)::int)
    );

-- ─── knowledge_entries RLS ────────────────────────────────────────────────
ALTER TABLE knowledge_entries ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS kn_admin_policy ON knowledge_entries;
CREATE POLICY kn_admin_policy ON knowledge_entries
    FOR ALL USING (current_setting('app.user_role', true) = 'Admin');

DROP POLICY IF EXISTS kn_dept_policy ON knowledge_entries;
CREATE POLICY kn_dept_policy ON knowledge_entries
    FOR ALL USING (
        visibility    = 'public'
        OR department_id = (current_setting('app.user_dept_id', true)::int)
        OR owner_id      = (current_setting('app.user_id',      true)::int)
    );

-- ─── document_versions RLS (migration 003) ────────────────────────────────
ALTER TABLE document_versions ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS docver_admin_policy ON document_versions;
CREATE POLICY docver_admin_policy ON document_versions
    FOR ALL USING (current_setting('app.user_role', true) = 'Admin');

-- ═══════════════════════════════════════════════════════════════════════════
-- SEED DATA — phòng ban mặc định
-- ═══════════════════════════════════════════════════════════════════════════
INSERT INTO departments (name, description) VALUES
    ('Kế toán',      'Phòng Kế toán - Tài chính'),
    ('Nhân sự',      'Phòng Nhân sự'),
    ('Kỹ thuật',     'Phòng Kỹ thuật - Công nghệ'),
    ('Pháp chế',     'Phòng Pháp chế - Hợp đồng'),
    ('Ban Giám đốc', 'Ban lãnh đạo')
ON CONFLICT (name) DO NOTHING;

-- ═══════════════════════════════════════════════════════════════════════════
DO $$ BEGIN
    RAISE NOTICE '✅ MiccoRAG-v3 schema ready (init + 002_rls + 003_versions + 004_doc_fields)';
END $$;
