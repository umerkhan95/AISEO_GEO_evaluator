"""
SQLite Database for GEO Optimization Jobs and Chunks.

Provides persistent storage for:
- Optimization jobs (website URLs, settings)
- Content chunks (sections of pages)
- Processing results (optimized content)
- Guidelines applied (audit trail)
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Database file location - use /data for persistent storage on Fly.io
import os
if os.path.exists("/data"):
    DB_PATH = Path("/data/geo_optimizer.db")
else:
    DB_PATH = Path(__file__).parent / "geo_optimizer.db"


@dataclass
class OptimizationJob:
    """Represents a website optimization job."""
    job_id: str
    url: str
    status: str  # pending, crawling, processing, completed, failed
    industry: Optional[str] = None
    industry_confidence: float = 0.0
    total_chunks: int = 0
    completed_chunks: int = 0
    original_geo_score: float = 0.0
    optimized_geo_score: float = 0.0
    created_at: str = ""
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    settings: str = "{}"  # JSON string
    crawl_metadata: str = "{}"  # JSON string with crawl stats


@dataclass
class ContentChunk:
    """Represents a chunk of content to be optimized."""
    chunk_id: str
    job_id: str
    chunk_order: int
    section_title: str
    original_content: str
    optimized_content: Optional[str] = None
    status: str = "pending"  # pending, processing, completed, failed
    geo_score_before: float = 0.0
    geo_score_after: float = 0.0
    guidelines_applied: str = "[]"  # JSON array of guideline IDs
    processing_time_ms: int = 0
    created_at: str = ""
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class GuidelineApplication:
    """Tracks which guidelines were applied to which chunks."""
    id: str
    chunk_id: str
    job_id: str
    guideline_id: str
    guideline_content: str
    guideline_category: str
    how_applied: str
    created_at: str = ""


def get_connection() -> sqlite3.Connection:
    """Get database connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def init_database():
    """Initialize database with required tables."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Jobs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_jobs (
                job_id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                industry TEXT,
                industry_confidence REAL DEFAULT 0.0,
                total_chunks INTEGER DEFAULT 0,
                completed_chunks INTEGER DEFAULT 0,
                original_geo_score REAL DEFAULT 0.0,
                optimized_geo_score REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                error_message TEXT,
                settings TEXT DEFAULT '{}',
                crawl_metadata TEXT DEFAULT '{}'
            )
        """)

        # Add crawl_metadata column if it doesn't exist (for existing DBs)
        try:
            cursor.execute("ALTER TABLE optimization_jobs ADD COLUMN crawl_metadata TEXT DEFAULT '{}'")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_chunks (
                chunk_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                chunk_order INTEGER NOT NULL,
                section_title TEXT NOT NULL,
                original_content TEXT NOT NULL,
                optimized_content TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                geo_score_before REAL DEFAULT 0.0,
                geo_score_after REAL DEFAULT 0.0,
                guidelines_applied TEXT DEFAULT '[]',
                processing_time_ms INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                error_message TEXT,
                FOREIGN KEY (job_id) REFERENCES optimization_jobs(job_id)
            )
        """)

        # Guidelines application tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS guideline_applications (
                id TEXT PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                guideline_id TEXT NOT NULL,
                guideline_content TEXT NOT NULL,
                guideline_category TEXT NOT NULL,
                how_applied TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES content_chunks(chunk_id),
                FOREIGN KEY (job_id) REFERENCES optimization_jobs(job_id)
            )
        """)

        # Final results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_results (
                result_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL UNIQUE,
                final_markdown TEXT NOT NULL,
                final_html TEXT,
                report_json TEXT NOT NULL,
                total_guidelines_applied INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (job_id) REFERENCES optimization_jobs(job_id)
            )
        """)

        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_job_id ON content_chunks(job_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_status ON content_chunks(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON optimization_jobs(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_guidelines_job ON guideline_applications(job_id)")

        print("✅ Database initialized successfully")


# ============================================================================
# Job Operations
# ============================================================================

def create_job(url: str, settings: Dict = None) -> str:
    """Create a new optimization job."""
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO optimization_jobs (job_id, url, status, created_at, settings)
            VALUES (?, ?, 'pending', ?, ?)
        """, (job_id, url, now, json.dumps(settings or {})))

    return job_id


def get_job(job_id: str) -> Optional[Dict]:
    """Get job by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM optimization_jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def update_job(job_id: str, **kwargs):
    """Update job fields."""
    if not kwargs:
        return

    fields = ", ".join(f"{k} = ?" for k in kwargs.keys())
    values = list(kwargs.values()) + [job_id]

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE optimization_jobs SET {fields} WHERE job_id = ?", values)


def get_all_jobs(limit: int = 50) -> List[Dict]:
    """Get all jobs ordered by creation date."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM optimization_jobs
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]


# ============================================================================
# Chunk Operations
# ============================================================================

def create_chunk(job_id: str, chunk_order: int, section_title: str, content: str) -> str:
    """Create a new content chunk."""
    chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO content_chunks
            (chunk_id, job_id, chunk_order, section_title, original_content, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (chunk_id, job_id, chunk_order, section_title, content, now))

    return chunk_id


def create_chunks_batch(job_id: str, chunks: List[Dict]) -> List[str]:
    """Create multiple chunks at once."""
    chunk_ids = []
    now = datetime.now().isoformat()

    with get_db() as conn:
        cursor = conn.cursor()
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk_{uuid.uuid4().hex[:12]}"
            cursor.execute("""
                INSERT INTO content_chunks
                (chunk_id, job_id, chunk_order, section_title, original_content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chunk_id, job_id, i, chunk.get("title", f"Section {i+1}"), chunk["content"], now))
            chunk_ids.append(chunk_id)

        # Update job with total chunks
        cursor.execute(
            "UPDATE optimization_jobs SET total_chunks = ? WHERE job_id = ?",
            (len(chunks), job_id)
        )

    return chunk_ids


def get_chunk(chunk_id: str) -> Optional[Dict]:
    """Get chunk by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM content_chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_job_chunks(job_id: str) -> List[Dict]:
    """Get all chunks for a job ordered by chunk_order."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM content_chunks
            WHERE job_id = ?
            ORDER BY chunk_order
        """, (job_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_pending_chunks(job_id: str) -> List[Dict]:
    """Get pending chunks for a job."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM content_chunks
            WHERE job_id = ? AND status = 'pending'
            ORDER BY chunk_order
        """, (job_id,))
        return [dict(row) for row in cursor.fetchall()]


def update_chunk(chunk_id: str, **kwargs):
    """Update chunk fields."""
    if not kwargs:
        return

    fields = ", ".join(f"{k} = ?" for k in kwargs.keys())
    values = list(kwargs.values()) + [chunk_id]

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE content_chunks SET {fields} WHERE chunk_id = ?", values)


def mark_chunk_completed(chunk_id: str, optimized_content: str, geo_score_after: float,
                         guidelines_applied: List[str], processing_time_ms: int):
    """Mark chunk as completed with results."""
    now = datetime.now().isoformat()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE content_chunks SET
                status = 'completed',
                optimized_content = ?,
                geo_score_after = ?,
                guidelines_applied = ?,
                processing_time_ms = ?,
                completed_at = ?
            WHERE chunk_id = ?
        """, (optimized_content, geo_score_after, json.dumps(guidelines_applied),
              processing_time_ms, now, chunk_id))

        # Get job_id and update completed count
        cursor.execute("SELECT job_id FROM content_chunks WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        if row:
            job_id = row["job_id"]
            cursor.execute("""
                UPDATE optimization_jobs SET completed_chunks = (
                    SELECT COUNT(*) FROM content_chunks
                    WHERE job_id = ? AND status = 'completed'
                ) WHERE job_id = ?
            """, (job_id, job_id))


# ============================================================================
# Guideline Application Tracking
# ============================================================================

def record_guideline_application(chunk_id: str, job_id: str, guideline_id: str,
                                  guideline_content: str, guideline_category: str,
                                  how_applied: str):
    """Record that a guideline was applied to a chunk."""
    record_id = f"ga_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO guideline_applications
            (id, chunk_id, job_id, guideline_id, guideline_content, guideline_category, how_applied, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (record_id, chunk_id, job_id, guideline_id, guideline_content, guideline_category, how_applied, now))


def get_job_guidelines_applied(job_id: str) -> List[Dict]:
    """Get all guidelines applied for a job."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM guideline_applications
            WHERE job_id = ?
            ORDER BY created_at
        """, (job_id,))
        return [dict(row) for row in cursor.fetchall()]


# ============================================================================
# Results Operations
# ============================================================================

def save_final_result(job_id: str, markdown: str, html: str, report: Dict, guidelines_count: int):
    """Save final optimization result."""
    result_id = f"result_{uuid.uuid4().hex[:12]}"
    now = datetime.now().isoformat()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO optimization_results
            (result_id, job_id, final_markdown, final_html, report_json, total_guidelines_applied, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (result_id, job_id, markdown, html, json.dumps(report), guidelines_count, now))

        # Update job as completed
        cursor.execute("""
            UPDATE optimization_jobs SET
                status = 'completed',
                completed_at = ?
            WHERE job_id = ?
        """, (now, job_id))


def get_result(job_id: str) -> Optional[Dict]:
    """Get final result for a job."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM optimization_results WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result["report_json"] = json.loads(result["report_json"])
            return result
        return None


# ============================================================================
# Statistics
# ============================================================================

def get_stats() -> Dict:
    """Get database statistics."""
    with get_db() as conn:
        cursor = conn.cursor()

        stats = {}

        # Job counts by status
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM optimization_jobs
            GROUP BY status
        """)
        stats["jobs_by_status"] = {row["status"]: row["count"] for row in cursor.fetchall()}

        # Total chunks
        cursor.execute("SELECT COUNT(*) as count FROM content_chunks")
        stats["total_chunks"] = cursor.fetchone()["count"]

        # Completed chunks
        cursor.execute("SELECT COUNT(*) as count FROM content_chunks WHERE status = 'completed'")
        stats["completed_chunks"] = cursor.fetchone()["count"]

        # Guidelines applied
        cursor.execute("SELECT COUNT(*) as count FROM guideline_applications")
        stats["total_guidelines_applied"] = cursor.fetchone()["count"]

        # Average scores
        cursor.execute("""
            SELECT
                AVG(original_geo_score) as avg_original,
                AVG(optimized_geo_score) as avg_optimized
            FROM optimization_jobs
            WHERE status = 'completed'
        """)
        row = cursor.fetchone()
        stats["avg_original_score"] = row["avg_original"] or 0
        stats["avg_optimized_score"] = row["avg_optimized"] or 0

        return stats


# Initialize on import
if __name__ == "__main__":
    init_database()
    print(f"Database location: {DB_PATH}")
