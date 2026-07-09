#!/usr/bin/env python3
"""
Labo Long-Term Memory — SQLite + Granite-97m Multilingual

Sistema de memória de longo prazo do Labo (Hermes Agent).
SQLite como source of truth + embeddings vetoriais via sqlite-vec.

Uso:
  query_memory.py add "Título" "Conteúdo" [--category cat] [--tags t1,t2]
  query_memory.py search "pergunta ou tema" [--top_k 5] [--category cat]
  query_memory.py get <id>
  query_memory.py update <id> [--title t] [--content c] [--category cat] [--tags t1,t2] [--status s]
  query_memory.py delete <id>
  query_memory.py list [--category cat] [--status ativa] [--limit 50]
  query_memory.py init                    # Criar schema
  query_memory.py import-vault <path>     # Importar notas .md do Obsidian
  query_memory.py backup                  # Dump SQL para stdout
  query_memory.py stats                   # Estatísticas do DB
  query_memory.py reindex                 # Reindexar todos os embeddings
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

# Silenciar warnings do HuggingFace
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
    """Conecta ao SQLite com WAL mode e sqlite-vec."""
    db = sqlite3.connect(DB_PATH)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.row_factory = sqlite3.Row

    # Carregar extensão sqlite-vec
    import sqlite_vec
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    return db


def init_db(db):
    """Cria tabelas e índices vetoriais."""
    db.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            category TEXT DEFAULT 'geral',
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

    # Criar tabela virtual vetorial se não existir
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

    # Criar índice FTS5 para busca lexical (BM25)
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

    # Migração one-time: se existem chunks mas FTS5 está vazio, reconstruir
    try:
        chunk_count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        fts_count = db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        if chunk_count > 0 and fts_count == 0:
            print(f"Migração FTS5: {chunk_count} chunks encontrados, reconstruindo índice...")
            _fts5_rebuild(db)
            print("Índice FTS5 reconstruído com sucesso.")
    except sqlite3.OperationalError:
        pass  # Tabela chunks ou chunks_fts ainda não existem

    # Registrar metadata
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
    print("Schema criado/verificado com sucesso.")


def iso_now():
    return datetime.datetime.now().isoformat()


import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Redirecionar stderr temporariamente durante import do embedder
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
    """Retorna instância singleton do GraniteONNXEmbedder."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = GraniteONNXEmbedder()
    return _MODEL_INSTANCE


def embed_texts(texts):
    """Gera embeddings para uma lista de textos usando Granite-97m ONNX int8."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True)


def embedding_to_blob(embedding):
    """Converte lista de floats para BLOB (float32 little-endian)."""
    if hasattr(embedding, 'tolist'):
        embedding = embedding.tolist()
    return struct.pack(f"<{len(embedding)}f", *embedding)


def vec_embedding(emb_list):
    """Converte lista de floats para JSON string aceito pelo sqlite-vec."""
    if hasattr(emb_list, 'tolist'):
        emb_list = emb_list.tolist()
    return json.dumps(emb_list)


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Divide texto longo em chunks com overlap."""
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
    """Sincroniza FTS5 após inserir um chunk."""
    try:
        db.execute(
            "INSERT INTO chunks_fts(rowid, chunk_text) VALUES (?, ?)",
            (chunk_id, chunk_text_content)
        )
    except sqlite3.OperationalError:
        pass  # FTS5 table may not exist during testing


def _fts5_delete(db, chunk_id):
    """Remove entrada do índice FTS5."""
    try:
        db.execute(
            "INSERT INTO chunks_fts(chunks_fts, rowid) VALUES('delete', ?)",
            (chunk_id,)
        )
    except sqlite3.OperationalError:
        pass


def _fts5_rebuild(db):
    """Reconstrói completo o índice FTS5 a partir da tabela chunks."""
    try:
        db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    except sqlite3.OperationalError:
        pass


def search_fts(db, query, top_k=5, category=None):
    """
    Busca lexical via FTS5 com ranking BM25.
    Retorna lista de dicts com memory_id, chunk_text, title, category, bm25_score.
    Se a query FTS5 falhar (syntax error), faz fallback para LIKE.
    """
    # Preparar query para FTS5: separar palavras, limpar, quote tokens especiais
    words = []
    for w in query.split():
        w = w.strip().strip(".,;:!?()[]{}""'")
        if len(w) < 2:
            continue
        # Tokens com caracteres especiais precisam de aspas duplas no FTS5
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
        # FTS5 query syntax error — fallback para LIKE
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
    Busca híbrida: semântica (sqlite-vec) + lexical (FTS5/BM25) com RRF merge.
    Retorna lista de dicts ordenada por RRF score descendente,
    cada dict com: memory_id, title, category, chunk_text, content,
    semantic_similarity, rrf_score.
    """
    overfetch = top_k * 3

    # ── 1. Busca semântica (sqlite-vec) ──
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

    # ── 2. Busca lexical (FTS5 / BM25) ──
    fts_results = search_fts(db, query, top_k, category)

    # ── 3. RRF Merge (Reciprocal Rank Fusion) ──
    K = 60  # Constante RRF padrão
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
            # Atualizar similaridade semântica se este chunk for melhor
            if sim > merged[mid]["semantic_similarity"]:
                merged[mid]["semantic_similarity"] = sim
        merged[mid]["rrf_score"] += 1.0 / (rank + K)

    for rank, r in enumerate(fts_results):
        mid = r["memory_id"]
        if mid not in merged:
            # Resultado veio só do FTS5 — buscar metadata completo
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


def add_memory(db, title, content, category="geral", tags="", source="labo"):
    """Adiciona uma memória com chunks e embeddings."""
    now = iso_now()

    # Inserir memória
    cur = db.execute(
        "INSERT INTO memories (title, category, content, tags, status, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (title, category, content, tags, "ativa", source, now, now)
    )
    memory_id = cur.lastrowid

    # Gerar chunks e embeddings
    chunks = chunk_text(content)
    if chunks:
        texts = [c[1] for c in chunks]
        embeddings = embed_texts(texts)

        for (chunk_idx, chunk_text_content), emb in zip(chunks, embeddings):
            blob = embedding_to_blob(emb)
            # Inserir chunk na tabela chunks
            chunk_cur = db.execute(
                "INSERT INTO chunks (memory_id, chunk_index, chunk_text, embedding) VALUES (?, ?, ?, ?)",
                (memory_id, chunk_idx, chunk_text_content, blob)
            )
            chunk_id = chunk_cur.lastrowid
            # Inserir no índice vetorial
            db.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, vec_embedding(emb))
            )
            # Sincronizar FTS5
            _fts5_insert(db, chunk_id, chunk_text_content)

    db.commit()
    print(f"Memória adicionada: id={memory_id}, title='{title}', category='{category}', chunks={len(chunks)}")
    return memory_id


def search_memory(db, query, top_k=5, category=None):
    """Busca híbrida: semântica + lexical (FTS5/BM25) com RRF merge. CLI-friendly."""
    results = hybrid_search(db, query, top_k, category)

    if not results:
        print("Nenhuma memória relevante encontrada.")
        return

    for i, r in enumerate(results, 1):
        sim = r.get("semantic_similarity", 0)
        rrf = r.get("rrf_score", 0)
        print(f"\n--- Resultado {i} (RRF: {rrf:.4f} | cos: {sim:.3f}) ---")
        print(f"ID: {r['memory_id']} | Título: {r['title']}")
        print(f"Categoria: {r['category']} | Tags: {r.get('tags', '')}")
        print(f"Trecho: {r['chunk_text'][:300]}...")
        if r.get("content") and sim > 0.5:
            print(f"\n[CONTEÚDO]:\n{r['content']}")


def get_memory(db, memory_id):
    """Recupera uma memória completa por ID."""
    row = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        print(f"Memória id={memory_id} não encontrada.")
        return
    print(f"\nID: {row['id']}")
    print(f"Título: {row['title']}")
    print(f"Categoria: {row['category']}")
    print(f"Tags: {row['tags']}")
    print(f"Status: {row['status']}")
    print(f"Fonte: {row['source']}")
    print(f"Criado: {row['created_at']}")
    print(f"Atualizado: {row['updated_at']}")
    print(f"\nConteúdo:\n{row['content']}")


def update_memory(db, memory_id, title=None, content=None, category=None, tags=None, status=None):
    """Atualiza uma memória existente."""
    row = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        print(f"Memória id={memory_id} não encontrada.")
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

    # Se conteúdo mudou, reindexar chunks
    if content and content != row["content"]:
        # Remover chunks antigos (incluindo FTS5)
        chunk_ids = db.execute("SELECT id FROM chunks WHERE memory_id = ?", (memory_id,)).fetchall()
        for cid in chunk_ids:
            db.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid[0],))
            _fts5_delete(db, cid[0])
        db.execute("DELETE FROM chunks WHERE memory_id = ?", (memory_id,))

        # Criar novos chunks
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
    print(f"Memória id={memory_id} atualizada.")


def delete_memory(db, memory_id):
    """Remove uma memória e seus chunks."""
    chunk_ids = db.execute("SELECT id FROM chunks WHERE memory_id = ?", (memory_id,)).fetchall()
    for cid in chunk_ids:
        db.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid[0],))
        _fts5_delete(db, cid[0])
    db.execute("DELETE FROM chunks WHERE memory_id = ?", (memory_id,))
    db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    db.commit()
    print(f"Memória id={memory_id} removida.")


def list_memories(db, category=None, status="ativa", limit=50):
    """Lista memórias com filtros."""
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
        print("Nenhuma memória encontrada.")
        return

    print(f"{'ID':<5} {'Categoria':<15} {'Status':<10} {'Atualizado':<20} {'Título'}")
    print("-" * 90)
    for r in rows:
        print(f"{r['id']:<5} {r['category']:<15} {r['status']:<10} {r['updated_at'][:16]:<20} {r['title']}")


def import_vault(db, vault_path):
    """Importa notas .md do Obsidian Vault para o SQLite."""
    vault = os.path.expanduser(vault_path)
    if not os.path.isdir(vault):
        print(f"Diretório não encontrado: {vault}")
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

        # Detectar categoria pelo nome
        category = "geral"
        if "Infraestrutura" in filename:
            category = "infraestrutura"
        elif "Pesquisa" in filename:
            category = "pesquisa"
        elif "Projeto" in filename or "Projetos" in filename:
            category = "projeto"
        elif "Config" in filename:
            category = "config"
        elif "Decis" in filename:
            category = "decisao"
        elif "Memória" in filename or "Memoria" in filename:
            category = "meta"
        elif "Trading" in filename:
            category = "projeto"

        # Verificar se já existe (dedup por título)
        existing = db.execute("SELECT id FROM memories WHERE title = ?", (filename,)).fetchone()
        if existing:
            print(f"  SKIP (já existe): {filename}")
            skipped += 1
            continue

        try:
            add_memory(db, filename, content, category=category, tags="imported-obsidian", source="obsidian-import")
            imported += 1
            print(f"  OK: {filename}")
        except Exception as e:
            print(f"  ERRO: {filename} — {e}")
            skipped += 1

    print(f"\nImportação concluída: {imported} importadas, {skipped} puladas.")


def backup_db(db):
    """Dump completo do banco em SQL texto para stdout."""
    import io
    output = io.StringIO()
    for line in db.iterdump():
        output.write(line + "\n")
    print(output.getvalue())


def stats_db(db):
    """Mostra estatísticas do banco."""
    mem_count = db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    chunk_count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    vec_count = db.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    categories = db.execute("SELECT category, COUNT(*) FROM memories GROUP BY category ORDER BY COUNT(*) DESC").fetchall()
    model = db.execute("SELECT value FROM metadata WHERE key = 'model'").fetchone()
    db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

    print(f"Banco: {DB_PATH}")
    print(f"Tamanho: {db_size / 1024:.1f} KB")
    print(f"Modelo: {model[0] if model else 'N/A'}")
    print(f"Memórias: {mem_count}")
    print(f"Chunks: {chunk_count}")
    print(f"Vetores indexados: {vec_count}")
    print(f"\nPor categoria:")
    for cat, count in categories:
        print(f"  {cat}: {count}")


def reindex_all(db):
    """Reindexa todos os embeddings do zero."""
    print("Reindexando todos os embeddings...")

    # Garantir que o schema FTS5 existe
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

    # Limpar índice vetorial
    db.execute("DELETE FROM vec_chunks")

    # Recriar tabela virtual (mais seguro)
    db.execute("DROP TABLE IF EXISTS vec_chunks")
    db.execute(f"""
        CREATE VIRTUAL TABLE vec_chunks
        USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding float[{EMBEDDING_DIM}]
        )
    """)

    # Regenerar chunks e embeddings para cada memória
    memories = db.execute("SELECT id, content FROM memories WHERE status = 'ativa'").fetchall()
    total = len(memories)

    # Limpar chunks existentes
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
            print(f"  {i + 1}/{total} memórias reindexadas")

    # Reconstruir índice FTS5
    _fts5_rebuild(db)
    print("Índice FTS5 reconstruído.")

    now = iso_now()
    db.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
               ("last_reindex", now))
    db.commit()
    print(f"Reindexação concluída: {total} memórias.")


def main():
    parser = argparse.ArgumentParser(description="Labo Long-Term Memory — SQLite + Granite-97m")
    sub = parser.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="Criar schema do banco")

    # add
    p_add = sub.add_parser("add", help="Adicionar memória")
    p_add.add_argument("title", help="Título da memória")
    p_add.add_argument("content", help="Conteúdo da memória")
    p_add.add_argument("--category", default="geral", help="Categoria")
    p_add.add_argument("--tags", default="", help="Tags separadas por vírgula")
    p_add.add_argument("--source", default="labo", help="Fonte da memória")

    # search
    p_search = sub.add_parser("search", help="Busca semântica")
    p_search.add_argument("query", help="Pergunta ou tema para buscar")
    p_search.add_argument("--top_k", type=int, default=5, help="Número de resultados")
    p_search.add_argument("--category", default=None, help="Filtrar por categoria")

    # get
    p_get = sub.add_parser("get", help="Recuperar memória por ID")
    p_get.add_argument("id", type=int, help="ID da memória")

    # update
    p_update = sub.add_parser("update", help="Atualizar memória")
    p_update.add_argument("id", type=int, help="ID da memória")
    p_update.add_argument("--title", default=None, help="Novo título")
    p_update.add_argument("--content", default=None, help="Novo conteúdo")
    p_update.add_argument("--category", default=None, help="Nova categoria")
    p_update.add_argument("--tags", default=None, help="Novas tags")
    p_update.add_argument("--status", default=None, help="Novo status")

    # delete
    p_del = sub.add_parser("delete", help="Remover memória")
    p_del.add_argument("id", type=int, help="ID da memória")

    # list
    p_list = sub.add_parser("list", help="Listar memórias")
    p_list.add_argument("--category", default=None, help="Filtrar por categoria")
    p_list.add_argument("--status", default="ativa", help="Filtrar por status")
    p_list.add_argument("--limit", type=int, default=50, help="Limite de resultados")

    # import-vault
    p_import = sub.add_parser("import-vault", help="Importar notas do Obsidian")
    p_import.add_argument("path", help="Caminho do Vault")

    # backup
    sub.add_parser("backup", help="Dump SQL para stdout")

    # stats
    sub.add_parser("stats", help="Estatísticas do banco")

    # reindex
    sub.add_parser("reindex", help="Reindexar todos os embeddings")

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
