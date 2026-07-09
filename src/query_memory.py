#!/usr/bin/env python3
"""
Labo Long-Term Memory — SQLite + Granite-97m Multilingual

Long-term memory system for Labo (Hermes Agent).
SQLite as source of truth + vector embeddings via sqlite-vec.

Usage:
  query_memory.py add "Title" "Content" [--category cat] [--tags t1,t2]
  query_memory.py search "query or topic" [--top_k 5] [--category cat]
  query_memory.py get <id>
  query_memory.py update <id> [--title t] [--content c] [--category cat] [--tags t1,t2] [--status s]
  query_memory.py delete <id>
  query_memory.py list [--category cat] [--status active] [--limit 50]
  query_memory.py init                    # Create DB schema
  query_memory.py import-vault <path>     # Import .md notes from Obsidian
  query_memory.py backup                  # SQL dump to stdout
  query_memory.py stats                   # DB statistics
  query_memory.py reindex                 # Reindex all embeddings
"""

import argparse
import datetime
import glob
import json
import os
import re
import sqlite3
import struct
import sys
import warnings

# Silence HuggingFace warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Config (override via env var LTM_DB_PATH for custom paths)
DB_PATH = os.environ.get("LTM_DB_PATH") or os.path.expanduser("~/.hermes/longterm-memory.db")
VENV_PYTHON = os.path.expanduser("~/.hermes/ltm-env/bin/python")
MODEL_NAME = "ibm-granite/granite-embedding-97m-multilingual-r2"
EMBEDDING_DIM = 384  # ModernBERT hidden_size
CHUNK_SIZE = 400  # chars per chunk for long content
CHUNK_OVERLAP = 50


def get_db():
    """Connect to SQLite with WAL mode and sqlite-vec."""
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.row_factory = sqlite3.Row

    # Load sqlite-vec extension
    import sqlite_vec
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    return db


def init_db(db):
    """Create tables and vector indexes."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            content TEXT NOT NULL,
            tags TEXT DEFAULT '',
            status TEXT DEFAULT 'ativa',
            source TEXT DEFAULT 'labo',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            chunk_text TEXT NOT NULL,
            embedding BLOB,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
        CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
        CREATE INDEX IF NOT EXISTS idx_chunks_memory_id ON chunks(memory_id);
    """)

    # Create vector virtual table if it doesn't exist
    try:
        db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks
            USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{EMBEDDING_DIM}]
            )
        """)
    except sqlite3.OperationalError:
        pass

    # Create FTS5 index for lexical search (BM25)
    try:
        db.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_text,
                content='chunks',
                content_rowid='id'
            );
        """)
    except sqlite3.OperationalError:
        pass

    # One-time migration: if chunks exist but FTS5 is empty, rebuild
    try:
        chunk_count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        fts_count = db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        if chunk_count > 0 and fts_count == 0:
            print(f"FTS5 migration: {chunk_count} chunks found, rebuilding index...")
            _fts5_rebuild(db)
            print("FTS5 index rebuilt successfully.")
    except sqlite3.OperationalError:
        pass  # chunks or chunks_fts table may not exist yet

    # Register metadata
    now = iso_now()
    db.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
               ("model", MODEL_NAME))
    db.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
               ("embedding_dim", str(EMBEDDING_DIM)))
    db.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
               ("created_at", now))
    db.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
               ("last_reindex", now))

    db.commit()
    print("Schema created/verified successfully.")

    # ── Category migration: translate legacy PT category names to EN ──
    CATEGORY_MIGRATION = {
        "geral": "general",
        "infraestrutura": "infrastructure",
        "projeto": "project",
        "pesquisa": "research",
        "decisao": "decision",
        "correcao": "correction",
    }
    for old_cat, new_cat in CATEGORY_MIGRATION.items():
        db.execute("UPDATE memories SET category = ? WHERE category = ?", (new_cat, old_cat))
    changed = db.execute("SELECT changes()").fetchone()[0]
    if changed > 0:
        print(f"Category migration: {changed} memories updated.")
    db.commit()


def iso_now():
    return datetime.datetime.now().isoformat()


import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Temporarily redirect stderr during embedder import
import sys as _sys
import io as _io
_old_stderr = _sys.stderr
_sys.stderr = _io.StringIO()
try:
    from granite_embedder import GraniteONNXEmbedder
finally:
    _sys.stderr = _old_stderr
    del _old_stderr, _io

# Singleton do modelo
_MODEL_INSTANCE = None

def _get_model():
    """Return singleton instance of GraniteONNXEmbedder."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = GraniteONNXEmbedder()
    return _MODEL_INSTANCE


def embed_texts(texts):
    """Generate embeddings for a list of texts using Granite-97m ONNX int8."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True)


def embedding_to_blob(embedding):
    """Convert a list of floats to a BLOB (float32 little-endian)."""
    if hasattr(embedding, 'tolist'):
        embedding = embedding.tolist()
    return struct.pack(f"<{len(embedding)}f", *embedding)


def vec_embedding(emb_list):
    """Convert a list of floats to a JSON string accepted by sqlite-vec."""
    if hasattr(emb_list, 'tolist'):
        emb_list = emb_list.tolist()
    return json.dumps(emb_list)


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split long text into chunks with overlap."""
    if len(text) <= size:
        return [(0, text)]

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append((idx, chunk))
        idx += 1
        start = end - overlap
        if start >= len(text):
            break
    return chunks


# ── FTS5 sync helpers ─────────────────────────────────────────────────────────

def _fts5_insert(db, chunk_id, chunk_text_content):
    """Sync FTS5 index after inserting a chunk."""
    try:
        db.execute(
            "INSERT INTO chunks_fts(rowid, chunk_text) VALUES (?, ?)",
            (chunk_id, chunk_text_content)
        )
    except sqlite3.OperationalError:
        pass  # FTS5 table may not exist during testing


def _fts5_delete(db, chunk_id):
    """Remove an entry from the FTS5 index."""
    try:
        db.execute(
            "INSERT INTO chunks_fts(chunks_fts, rowid) VALUES('delete', ?)",
            (chunk_id,)
        )
    except sqlite3.OperationalError:
        pass


def _fts5_rebuild(db):
    """Rebuild the entire FTS5 index from the chunks table."""
    try:
        db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    except sqlite3.OperationalError:
        pass


def search_fts(db, query, top_k=5, category=None):
    """
    Lexical search via FTS5 with BM25 ranking.
    Returns a list of dicts with memory_id, chunk_text, title, category, bm25_score.
    Falls back to LIKE if the FTS5 query fails (syntax error).
    """
    # Prepare query for FTS5: split words, clean, quote special tokens
    words = []
    for w in query.split():
        w = w.strip().strip(".,;:!?()[]{}""'")
        if len(w) < 2:
            continue
        # Tokens with special characters need double quotes in FTS5
        if any(c in w for c in ":-_.@/\\"):
            words.append(f'"{w}"')
        else:
            words.append(w)

    fts_query = " AND ".join(words) if words else ""
    if not fts_query:
        return []

    sql = """
        SELECT
            c.memory_id,
            c.chunk_text,
            m.title,
            m.category,
            rank as bm25_score
        FROM chunks_fts
        JOIN chunks c ON chunks_fts.rowid = c.id
        JOIN memories m ON c.memory_id = m.id
        WHERE chunks_fts MATCH ?
          AND m.status = 'ativa'
    """
    params = [fts_query]
    if category:
        sql += " AND m.category = ?"
        params.append(category)
    sql += " ORDER BY rank LIMIT ?"
    params.append(top_k * 2)

    try:
        results = db.execute(sql, params).fetchall()
        return [dict(r) for r in results]
    except sqlite3.OperationalError:
        # FTS5 query syntax error — fallback to LIKE
        like_query = "%" + query.replace("%", "%%").replace("_", "\\_") + "%"
        fb_sql = """
            SELECT c.memory_id, c.chunk_text, m.title, m.category, 0 as bm25_score
            FROM chunks c
            JOIN memories m ON c.memory_id = m.id
            WHERE c.chunk_text LIKE ?
              AND m.status = 'ativa'
        """
        fb_params = [like_query]
        if category:
            fb_sql += " AND m.category = ?"
            fb_params.append(category)
        fb_sql += " LIMIT ?"
        fb_params.append(top_k * 2)
        results = db.execute(fb_sql, fb_params).fetchall()
        return [dict(r) for r in results]


def hybrid_search(db, query, top_k=5, category=None):
    """
    Hybrid search: semantic (sqlite-vec) + lexical (FTS5/BM25) with RRF merge.
    Returns a list of dicts ordered by descending RRF score,
    each dict with: memory_id, title, category, chunk_text, content,
    semantic_similarity, rrf_score.
    """
    overfetch = top_k * 3

    # ── 1. Semantic search (sqlite-vec) ──
    query_emb = embed_texts([query])[0]
    vec_sql = """
        SELECT
            vec.chunk_id, vec.distance,
            c.memory_id, c.chunk_text,
            m.title, m.category, m.tags, m.content, m.updated_at
        FROM vec_chunks vec
        JOIN chunks c ON vec.chunk_id = c.id
        JOIN memories m ON c.memory_id = m.id
        WHERE m.status = 'ativa'
          AND vec.embedding MATCH ?
          AND k = ?
    """
    vec_params = [vec_embedding(query_emb), overfetch]
    if category:
        vec_sql += " AND m.category = ?"
        vec_params.append(category)
    vec_sql += " ORDER BY vec.distance ASC"
    vec_results = db.execute(vec_sql, vec_params).fetchall()

    # ── 2. Lexical search (FTS5 / BM25) ──
    fts_results = search_fts(db, query, top_k, category)

    # ── 3. RRF Merge (Reciprocal Rank Fusion) ──
    K = 60  # Standard RRF constant
    merged = {}  # memory_id -> entry

    for rank, r in enumerate(vec_results):
        mid = r["memory_id"]
        sim = 1 - (r["distance"] ** 2) / 2  # dist euclidiana -> cosseno
        if mid not in merged:
            merged[mid] = {
                "memory_id": mid,
                "chunk_text": r["chunk_text"],
                "title": r["title"],
                "category": r["category"],
                "tags": r["tags"],
                "content": r["content"],
                "updated": r["updated_at"],
                "semantic_similarity": sim,
                "rrf_score": 0.0,
            }
        else:
            # Update semantic similarity if this chunk is better
            if sim > merged[mid]["semantic_similarity"]:
                merged[mid]["semantic_similarity"] = sim
        merged[mid]["rrf_score"] += 1.0 / (rank + K)

    for rank, r in enumerate(fts_results):
        mid = r["memory_id"]
        if mid not in merged:
            # Result came only from FTS5 — fetch full metadata
            full = db.execute(
                "SELECT content, tags, updated_at FROM memories WHERE id = ?",
                (mid,)
            ).fetchone()
            merged[mid] = {
                "memory_id": mid,
                "chunk_text": r["chunk_text"],
                "title": r["title"],
                "category": r["category"],
                "tags": full["tags"] if full else "",
                "content": full["content"] if full else r["chunk_text"],
                "updated": full["updated_at"] if full else "",
                "semantic_similarity": 0.0,
                "rrf_score": 0.0,
            }
        merged[mid]["rrf_score"] += 1.0 / (rank + K)

    # Ordenar por RRF score descendente e limitar
    ranked = sorted(merged.values(), key=lambda x: -x["rrf_score"])[:top_k]
    return ranked


def add_memory(db, title, content, category="general", tags="", source="labo"):
    """Add a memory entry with chunks and embeddings."""
    now = iso_now()

    # Insert memory
    cur = db.execute(
        "INSERT INTO memories (title, category, content, tags, status, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (title, category, content, tags, "ativa", source, now, now)
    )
    memory_id = cur.lastrowid

    # Generate chunks and embeddings
    chunks = chunk_text(content)
    if chunks:
        texts = [c[1] for c in chunks]
        embeddings = embed_texts(texts)

        for (chunk_idx, chunk_text_content), emb in zip(chunks, embeddings):
            blob = embedding_to_blob(emb)
            # Insert chunk into chunks table
            chunk_cur = db.execute(
                "INSERT INTO chunks (memory_id, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?)",
                (memory_id, chunk_idx, chunk_text_content, blob)
            )
            chunk_id = chunk_cur.lastrowid
            # Insert into vector index
            db.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, vec_embedding(emb))
            )
            # Sync FTS5
            _fts5_insert(db, chunk_id, chunk_text_content)

    db.commit()
    print(f"Memory added: id={memory_id}, title='{title}', category='{category}', chunks={len(chunks)}")
    return memory_id


def search_memory(db, query, top_k=5, category=None):
    """Hybrid search: semantic + lexical (FTS5/BM25) with RRF merge. CLI-friendly output."""
    results = hybrid_search(db, query, top_k, category)

    if not results:
        print("No relevant memories found.")
        return

    for i, r in enumerate(results, 1):
        sim = r.get("semantic_similarity", 0)
        rrf = r.get("rrf_score", 0)
        print(f"\n--- Result {i} (RRF: {rrf:.4f} | cos: {sim:.3f}) ---")
        print(f"ID: {r['memory_id']} | Title: {r['title']}")
        print(f"Category: {r['category']} | Tags: {r.get('tags', '')}")
        print(f"Snippet: {r['chunk_text'][:300]}...")
        if r.get("content") and sim > 0.5:
            print(f"\n[CONTENT]:\n{r['content']}")


def get_memory(db, memory_id):
    """Retrieve a complete memory entry by ID."""
    row = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        print(f"Memory id={memory_id} not found.")
        return
    print(f"\nID: {row['id']}")
    print(f"Title: {row['title']}")
    print(f"Category: {row['category']}")
    print(f"Tags: {row['tags']}")
    print(f"Status: {row['status']}")
    print(f"Source: {row['source']}")
    print(f"Created: {row['created_at']}")
    print(f"Updated: {row['updated_at']}")
    print(f"\nContent:\n{row['content']}")


def update_memory(db, memory_id, title=None, content=None, category=None, tags=None, status=None):
    """Update an existing memory entry."""
    row = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        print(f"Memory id={memory_id} not found.")
        return

    now = iso_now()
    new_title = title if title else row["title"]
    new_category = category if category else row["category"]
    new_tags = tags if tags else row["tags"]
    new_status = status if status else row["status"]
    new_content = content if content else row["content"]

    db.execute("""
        UPDATE memories SET title=?, category=?, tags=?, status=?, content=?, updated_at=?
        WHERE id=?
    """, (new_title, new_category, new_tags, new_status, new_content, now, memory_id))

    # If content changed, reindex chunks
    if content and content != row["content"]:
        # Remove old chunks (including FTS5)
        chunk_ids = db.execute("SELECT id FROM chunks WHERE memory_id = ?", (memory_id,)).fetchall()
        for cid in chunk_ids:
            db.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid[0],))
            _fts5_delete(db, cid[0])
        db.execute("DELETE FROM chunks WHERE memory_id = ?", (memory_id,))

        # Create new chunks
        chunks = chunk_text(content)
        if chunks:
            texts = [c[1] for c in chunks]
            embeddings = embed_texts(texts)
            for (chunk_idx, chunk_text_content), emb in zip(chunks, embeddings):
                blob = embedding_to_blob(emb)
                chunk_cur = db.execute(
                    "INSERT INTO chunks (memory_id, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?)",
                    (memory_id, chunk_idx, chunk_text_content, blob)
                )
                chunk_id = chunk_cur.lastrowid
                db.execute(
                    "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, vec_embedding(emb))
                )
                _fts5_insert(db, chunk_id, chunk_text_content)

    db.commit()
    print(f"Memory id={memory_id} updated.")


def delete_memory(db, memory_id):
    """Remove a memory entry and its chunks."""
    chunk_ids = db.execute("SELECT id FROM chunks WHERE memory_id = ?", (memory_id,)).fetchall()
    for cid in chunk_ids:
        db.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid[0],))
        _fts5_delete(db, cid[0])
    db.execute("DELETE FROM chunks WHERE memory_id = ?", (memory_id,))
    db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    db.commit()
    print(f"Memory id={memory_id} removed.")


def list_memories(db, category=None, status="ativa", limit=50):
    """List memories with filters."""
    sql = "SELECT id, title, category, tags, status, updated_at FROM memories WHERE 1=1"
    params = []
    if category:
        sql += " AND category = ?"
        params.append(category)
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)

    rows = db.execute(sql, params).fetchall()
    if not rows:
        print("No memories found.")
        return

    print(f"{'ID':<5} {'Category':<15} {'Status':<10} {'Updated':<20} {'Title'}")
    print("-" * 90)
    for r in rows:
        print(f"{r['id']:<5} {r['category']:<15} {r['status']:<10} {r['updated_at'][:16]:<20} {r['title']}")


def import_vault(db, vault_path):
    """Import .md notes from Obsidian Vault into SQLite."""
    vault = os.path.expanduser(vault_path)
    if not os.path.isdir(vault):
        print(f"Directory not found: {vault}")
        return

    md_files = glob.glob(os.path.join(vault, "**", "*.md"), recursive=True)
    md_files = [f for f in md_files if "/.obsidian/" not in f]

    imported = 0
    skipped = 0

    for fpath in sorted(md_files):
        filename = os.path.basename(fpath).replace(".md", "")
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content or len(content) < 30:
            skipped += 1
            continue

        # Detect category by filename (supports PT and EN naming)
        category = "general"
        if any(kw in filename for kw in ["Infraestrutura", "Infrastructure", "Infra", "Server", "Deploy"]):
            category = "infrastructure"
        elif any(kw in filename for kw in ["Pesquisa", "Research", "Study", "Library"]):
            category = "research"
        elif any(kw in filename for kw in ["Projeto", "Projetos", "Project", "App", "Feature"]):
            category = "project"
        elif "Config" in filename:
            category = "config"
        elif any(kw in filename for kw in ["Decis", "Decision", "Decisão", "Trade-off"]):
            category = "decision"
        elif any(kw in filename for kw in ["Memória", "Memoria", "Memory", "Meta"]):
            category = "meta"
        elif "Trading" in filename:
            category = "project"

        # Check if already exists (dedup by title)
        existing = db.execute("SELECT id FROM memories WHERE title = ?", (filename,)).fetchone()
        if existing:
            print(f"  SKIP (already exists): {filename}")
            skipped += 1
            continue

        try:
            add_memory(db, filename, content, category=category, tags="imported-obsidian", source="obsidian-import")
            imported += 1
            print(f"  OK: {filename}")
        except Exception as e:
            print(f"  ERROR: {filename} — {e}")
            skipped += 1

    print(f"\nImport complete: {imported} imported, {skipped} skipped.")


def backup_db(db):
    """Dump the complete database as SQL text to stdout."""
    import io
    output = io.StringIO()
    for line in db.iterdump():
        output.write(line + "\n")
    print(output.getvalue())


def stats_db(db):
    """Show database statistics."""
    mem_count = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    chunk_count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    vec_count = db.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    categories = db.execute("SELECT category, COUNT(*) FROM memories GROUP BY category ORDER BY COUNT(*) DESC").fetchall()
    model = db.execute("SELECT value FROM metadata WHERE key = 'model'").fetchone()
    db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

    print(f"Database: {DB_PATH}")
    print(f"Size: {db_size / 1024:.1f} KB")
    print(f"Model: {model[0] if model else 'N/A'}")
    print(f"Memories: {mem_count}")
    print(f"Chunks: {chunk_count}")
    print(f"Indexed vectors: {vec_count}")
    print(f"\nBy category:")
    for cat, count in categories:
        print(f"  {cat}: {count}")


def reindex_all(db):
    """Reindex all embeddings from scratch."""
    print("Reindexing all embeddings...")

    # Ensure FTS5 schema exists
    try:
        db.executescript("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_text,
                content='chunks',
                content_rowid='id'
            );
        """)
    except sqlite3.OperationalError:
        pass

    # Clear vector index
    db.execute("DELETE FROM vec_chunks")

    # Recreate virtual table (safer)
    db.execute("DROP TABLE IF EXISTS vec_chunks")
    db.execute(f"""
        CREATE VIRTUAL TABLE vec_chunks
        USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}]
        )
    """)

    # Regenerate chunks and embeddings for each memory
    memories = db.execute("SELECT id, content FROM memories WHERE status = 'ativa'").fetchall()
    total = len(memories)

    # Clear existing chunks
    db.execute("DELETE FROM chunks")

    for i, mem in enumerate(memories):
        memory_id = mem["id"]
        content = mem["content"]
        chunks = chunk_text(content)

        if chunks:
            texts = [c[1] for c in chunks]
            embeddings = embed_texts(texts)

            for (chunk_idx, chunk_text_content), emb in zip(chunks, embeddings):
                blob = embedding_to_blob(emb)
                chunk_cur = db.execute(
                    "INSERT INTO chunks (memory_id, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?)",
                    (memory_id, chunk_idx, chunk_text_content, blob)
                )
                chunk_id = chunk_cur.lastrowid
                db.execute(
                    "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, vec_embedding(emb))
                )

        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"  {i + 1}/{total} memories reindexed")

    # Rebuild FTS5 index
    _fts5_rebuild(db)
    print("FTS5 index rebuilt.")

    now = iso_now()
    db.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
               ("last_reindex", now))
    db.commit()
    print(f"Reindex complete: {total} memories.")


def main():
    parser = argparse.ArgumentParser(description="Labo Long-Term Memory — SQLite + Granite-97m")
    sub = parser.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="Create DB schema")

    # add
    p_add = sub.add_parser("add", help="Add memory")
    p_add.add_argument("title", help="Memory title")
    p_add.add_argument("content", help="Memory content")
    p_add.add_argument("--category", default="general", help="Category")
    p_add.add_argument("--tags", default="", help="Comma-separated tags")
    p_add.add_argument("--source", default="labo", help="Memory source")

    # search
    p_search = sub.add_parser("search", help="Semantic search")
    p_search.add_argument("query", help="Query topic to search for")
    p_search.add_argument("--top_k", type=int, default=5, help="Number of results")
    p_search.add_argument("--category", default=None, help="Filter by category")

    # get
    p_get = sub.add_parser("get", help="Retrieve memory by ID")
    p_get.add_argument("id", type=int, help="Memory ID")

    # update
    p_update = sub.add_parser("update", help="Update memory")
    p_update.add_argument("id", type=int, help="Memory ID")
    p_update.add_argument("--title", default=None, help="New title")
    p_update.add_argument("--content", default=None, help="New content")
    p_update.add_argument("--category", default=None, help="New category")
    p_update.add_argument("--tags", default=None, help="New tags")
    p_update.add_argument("--status", default=None, help="New status")

    # delete
    p_del = sub.add_parser("delete", help="Remove memory")
    p_del.add_argument("id", type=int, help="Memory ID")

    # list
    p_list = sub.add_parser("list", help="List memories")
    p_list.add_argument("--category", default=None, help="Filter by category")
    p_list.add_argument("--status", default="ativa", help="Filter by status")
    p_list.add_argument("--limit", type=int, default=50, help="Result limit")

    # import-vault
    p_import = sub.add_parser("import-vault", help="Import notes from Obsidian")
    p_import.add_argument("path", help="Vault path")

    # backup
    sub.add_parser("backup", help="SQL dump to stdout")

    # stats
    sub.add_parser("stats", help="Database statistics")

    # reindex
    sub.add_parser("reindex", help="Reindex all embeddings")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    db = get_db()

    if args.command == "init":
        init_db(db)

    elif args.command == "add":
        add_memory(db, args.title, args.content, args.category, args.tags, args.source)

    elif args.command == "search":
        search_memory(db, args.query, args.top_k, args.category)

    elif args.command == "get":
        get_memory(db, args.id)

    elif args.command == "update":
        update_memory(db, args.id, args.title, args.content, args.category, args.tags, args.status)

    elif args.command == "delete":
        delete_memory(db, args.id)

    elif args.command == "list":
        list_memories(db, args.category, args.status, args.limit)

    elif args.command == "import-vault":
        import_vault(db, args.path)

    elif args.command == "backup":
        backup_db(db)

    elif args.command == "stats":
        stats_db(db)

    elif args.command == "reindex":
        reindex_all(db)

    db.close()


if __name__ == "__main__":
    main()
