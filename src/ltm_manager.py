#!/usr/bin/env python3
"""
Labo Long-Term Memory Manager — Hermes <-> SQLite + Granite-97m

High-level layer used by the Hermes Agent to:
  1. OFFLOAD: migrate runtime memory entries into SQLite (frees up context)
  2. QUERY: hybrid semantic + lexical search by session topic
  3. CONSOLIDATE: dedup + merge of similar memories
  4. SYNC: keep runtime memory lean with pointers

Usage:
  ltm_manager.py offload <json_entries>     # Receive JSON from runtime, archive in SQLite
  ltm_manager.py query <topic> [--top_k 5]  # Semantic search by topic
  ltm_manager.py context <topics_json>       # Generate relevant context for prompt injection
  ltm_manager.py consolidate [--threshold 0.85]  # Dedup of similar memories
  ltm_manager.py status                      # Report: runtime vs SQLite
  ltm_manager.py export-obsidian <path>      # Export everything to Obsidian format
  ltm_manager.py init-session                # Return relevant context for new session
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import datetime

# Reuse query_memory infrastructure
sys.path.insert(0, os.path.dirname(__file__))
from query_memory import (
    DB_PATH, VENV_PYTHON, MODEL_NAME, EMBEDDING_DIM,
    get_db, init_db, embed_texts, chunk_text,
    embedding_to_blob, vec_embedding, iso_now,
    add_memory, search_memory as _search_memory,
    hybrid_search,
    get_memory, update_memory, delete_memory, list_memories,
    stats_db
)

# ---------------------------------------------------------------------------
# Multilingual seed phrases for centroid-based category classification
#
# Each category has seed phrases in multiple languages (PT, EN, ES, FR).
# The Granite-97m multilingual embedding model projects them into a shared
# semantic space, and the centroid (mean embedding) is used for classification.
# This works for ANY language the model supports — no keyword lists needed.
# ---------------------------------------------------------------------------
CATEGORY_SEEDS = {
    "general": [
        "General notes and miscellaneous information",
        "Notas gerais e informações diversas",
        "Uncategorized content and random notes",
        "Notes générales et informations diverses",
    ],
    "infrastructure": [
        "Server setup, deployment and networking",
        "Configuração de servidor, docker e infraestrutura",
        "SSH, nginx, firewall and system administration",
        "Implantación de servicios y administración de redes",
    ],
    "config": [
        "Configuration files, environment variables and settings",
        "Configurações de YAML, variáveis de ambiente e setup",
        "Installation and setup instructions",
        "Fichiers de configuration, variables d'environnement",
    ],
    "project": [
        "Project development, features and modules",
        "Desenvolvimento de projeto, bot, app e sistema",
        "Pipeline implementation and application architecture",
        "Desarrollo de proyectos y aplicaciones",
    ],
    "research": [
        "Technical research, frameworks and benchmarks",
        "Pesquisa de bibliotecas, APIs e referências técnicas",
        "Comparison of tools, SDKs and technologies",
        "Recherche technique, frameworks et API",
    ],
    "decision": [
        "Architectural decisions, trade-offs and technical debates",
        "Decisões de arquitetura, design rationale e trade-offs",
        "Technical choices and approach justifications",
        "Decisiones técnicas y justificaciones de diseño",
        "Choosing between PostgreSQL and MySQL for the database",
        "Opting for Redis instead of Memcached for caching",
        "Decidir entre FastAPI e Flask para a API",
        "We chose React over Vue for the frontend framework",
        "Evaluating and selecting cloud providers or hosting solutions",
    ],
    "correction": [
        "Rules, constraints and mandatory requirements",
        "Regras importantes, correções e proibições",
        "Never-do patterns and behavioral guidelines",
        "Règles importantes et comportements à éviter",
        "Nunca instale pacotes sem verificar a procedência",
        "Never commit secrets or credentials to the repository",
        "Always validate input before processing user data",
        "Obrigatório usar type hints em todas as funções",
        "Must follow the established coding conventions",
    ],
    "debug": [
        "Bug fixes, errors, workarounds and troubleshooting",
        "Correção de bugs, erros, timeout e debug",
        "Failure analysis and crash resolution",
        "Analyse d'erreurs, correctifs et dépannage",
    ],
}

# Module-level cache for centroid embeddings — built lazily on first classify_category() call
_CATEGORY_CENTROIDS = {}


def _build_category_centroids():
    """Build centroid embeddings for each category using the Granite multilingual model.

    Embeds all seed phrases for all categories in a single batch call, then
    averages per-category embeddings to produce one centroid per category.
    Centroids are cached globally and rebuilt on subprocess restart.
    """
    global _CATEGORY_CENTROIDS
    if _CATEGORY_CENTROIDS:
        return  # Already built

    # Flatten seeds, keeping track of category boundaries
    boundaries = []
    all_seeds = []
    for cat, seeds in CATEGORY_SEEDS.items():
        if not seeds:
            continue
        boundaries.append((cat, len(seeds)))
        all_seeds.extend(seeds)

    if not all_seeds:
        return

    try:
        all_embs = embed_texts(all_seeds)  # Single batch call
    except Exception:
        # Model not available — centroids stay empty, classify_category returns "general"
        return

    idx = 0
    for cat, count in boundaries:
        cat_embs = all_embs[idx:idx + count]
        # Average all seed embeddings to get centroid
        centroid = [sum(vals) / count for vals in zip(*cat_embs)]
        _CATEGORY_CENTROIDS[cat] = centroid
        idx += count


def _cosine_similarity(a, b):
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-10)


def classify_category(text):
    """Classify text into a category using embedding similarity.

    Uses the Granite multilingual model to compute semantic similarity
    between the input text and each category's centroid. This approach
    works across ALL languages supported by the embedding model —
    no hardcoded keyword lists per language.

    Falls back to 'general' when confidence is below 0.15 threshold
    or when the embedding model is unavailable.
    """
    if not text or not text.strip():
        return "general"

    # Build centroids lazily on first call
    if not _CATEGORY_CENTROIDS:
        _build_category_centroids()

    # If centroids failed to build (model unavailable), return fallback
    if not _CATEGORY_CENTROIDS:
        return "general"

    try:
        emb = embed_texts([text])[0]
    except Exception:
        return "general"

    best_cat = "general"
    best_sim = -1.0
    threshold = 0.15

    for cat, centroid in _CATEGORY_CENTROIDS.items():
        if centroid is None:
            continue
        sim = _cosine_similarity(emb, centroid)
        if sim > best_sim:
            best_sim = sim
            best_cat = cat

    return best_cat if best_sim >= threshold else "general"


# Entries that should NOT be offloaded (stay in runtime memory)
RUNTIME_ONLY_PATTERNS = [
    r"call.*Labo",
    r"respostas concisas",
    r"concise answers",
    r"NUNCA instalar",
    r"NEVER install",
    r"NUNCA criar cron",
    r"NEVER create cron",
    r"modelo padrão",
    r"default model",
    r"StealthyFetcher",
    r"prefere",
    r"prefer",
    r"Navarro",
]


def should_stay_runtime(text):
    """Check if memory entry should remain in runtime memory instead of SQLite.

    Returns True for entries matching patterns that are session-specific
    or private (user preferences, behavioral rules, personal identifiers).
    """
    for pattern in RUNTIME_ONLY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def offload_entries(entries_json):
    """Receive JSON runtime entries and archive them in SQLite.

    Returns JSON with: offloaded (successful), skipped (stay in runtime), errors.

    Input format: [{"text": "...", "target": "memory"|"user"}, ...]
    """
    try:
        entries = json.loads(entries_json) if isinstance(entries_json, str) else entries_json
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    db = get_db()
    # Ensure schema
    try:
        init_db(db)
    except Exception:
        pass  # Schema already exists

    result = {"offloaded": [], "skipped": [], "errors": []}

    for entry in entries:
        text = entry.get("text", "").strip()
        target = entry.get("target", "memory")

        if not text or len(text) < 20:
            result["skipped"].append({"reason": "too_short", "text": text[:80]})
            continue

        # Check if it should stay in runtime
        if should_stay_runtime(text):
            result["skipped"].append({"reason": "runtime_essential", "text": text[:80]})
            continue

        # Classify category
        category = classify_category(text)

        # Generate auto title
        title = generate_title(text, target)

        # Check dedup by similarity (search before insert)
        try:
            dup_check = check_duplicate(db, text, threshold=0.88)
            if dup_check:
                result["skipped"].append({
                    "reason": "duplicate",
                    "existing_id": dup_check,
                    "text": text[:80]
                })
                continue
        except Exception:
            pass  # If search fails, proceed with insertion

        # Insert into SQLite
        try:
            mem_id = add_memory(db, title, text, category=category,
                              tags=f"offload-{target}", source=f"hermes-runtime-{target}")
            result["offloaded"].append({
                "id": mem_id,
                "title": title,
                "category": category,
                "text_preview": text[:100]
            })
        except Exception as e:
            result["errors"].append({"text": text[:80], "error": str(e)})

    db.close()
    return json.dumps(result, ensure_ascii=False)


def check_duplicate(db, text, threshold=0.88):
    """Check if a similar memory already exists. Returns ID if duplicate, None otherwise."""
    try:
        query_emb = embed_texts([text])[0]
        k = 3
        sql = """
        SELECT vec.chunk_id, vec.distance, c.memory_id
        FROM vec_chunks vec
        JOIN chunks c ON vec.chunk_id = c.id
        JOIN memories m ON c.memory_id = m.id
        WHERE m.status = 'ativa'
        AND vec.embedding MATCH ?
        AND k = ?
        ORDER BY vec.distance ASC
        """
        results = db.execute(sql, [vec_embedding(query_emb), k]).fetchall()
        if results:
            # Convert euclidean distance to cosine similarity
            similarity = 1 - (results[0]["distance"] ** 2) / 2
            if similarity >= threshold:
                return results[0]["memory_id"]
    except Exception:
        pass
    return None


def generate_title(text, target="memory"):
    """Auto-generate a title for the entry."""
    # Get first meaningful line
    lines = text.strip().split("\n")
    first_line = ""
    for line in lines:
        clean = line.strip().lstrip("•-*§→").strip()
        if clean and len(clean) > 5:
            first_line = clean
            break

    if not first_line:
        first_line = text[:60]

    # Truncate
    title = first_line[:80]
    if len(first_line) > 80:
        title = title.rsplit(" ", 1)[0] + "..."

    # Prefix by target
    prefix = "User" if target == "user" else "Labo"
    return f"{prefix} — {title}"


def query_context(topic, top_k=5, category=None):
    """Hybrid search (semantic + lexical FTS5/BM25) returning JSON context.

    Used by Hermes to inject relevant context into the prompt.
    """
    db = get_db()

    try:
        results = hybrid_search(db, topic, top_k, category)
    except Exception as e:
        db.close()
        return json.dumps({"error": f"Hybrid search error: {e}"})

    context_entries = []
    for r in results:
        sim = r.get("semantic_similarity", 0)
        rrf = r.get("rrf_score", 0)

        # Less restrictive threshold for results that came from FTS5
        if sim < 0.30 and rrf < 0.008:
            continue

        context_entries.append({
            "id": r["memory_id"],
            "title": r["title"],
            "category": r["category"],
            "tags": r.get("tags", ""),
            "similarity": round(sim, 3),
            "rrf_score": round(rrf, 4),
            "updated": r.get("updated", ""),
            "relevant_chunk": r.get("chunk_text", "")[:500],
            "full_content": r.get("content") if sim > 0.45 else None,
        })

        if len(context_entries) >= top_k:
            break

    db.close()
    return json.dumps({
        "query": topic,
        "results_count": len(context_entries),
        "entries": context_entries
    }, ensure_ascii=False)


def init_session_context():
    """Return a summarized context for session startup.

    Combines: DB stats + top categories + latest updates.
    """
    db = get_db()

    # Overall stats
    mem_count = db.execute("SELECT COUNT(*) FROM memories WHERE status='ativa'").fetchone()[0]

    # Categories
    cats = db.execute(
        "SELECT category, COUNT(*) as c FROM memories WHERE status='ativa' GROUP BY category ORDER BY c DESC"
    ).fetchall()

    # Last 5 updates
    recent = db.execute(
        "SELECT id, title, category, updated_at FROM memories WHERE status='ativa' ORDER BY updated_at DESC LIMIT 5"
    ).fetchall()

    db.close()

    result = {
        "total_memories": mem_count,
        "categories": {r["category"]: r["c"] for r in cats},
        "recent_updates": [
            {"id": r["id"], "title": r["title"], "category": r["category"], "updated": r["updated_at"]}
            for r in recent
        ],
        "db_path": DB_PATH,
    }
    return json.dumps(result, ensure_ascii=False)


def consolidate_memories(threshold=0.85):
    """Find similar memories and propose a merge.

    Returns JSON with duplicate pairs for approval.
    """
    db = get_db()
    memories = db.execute(
        "SELECT id, title, content, category FROM memories WHERE status='ativa' ORDER BY id"
    ).fetchall()

    if len(memories) < 2:
        db.close()
        return json.dumps({"message": "Not enough memories to consolidate", "pairs": []})

    # Compare all pairs via embeddings
    texts = [f"{m['title']} {m['content'][:200]}" for m in memories]

    try:
        embeddings = embed_texts(texts)
    except Exception as e:
        db.close()
        return json.dumps({"error": f"Error generating embeddings: {e}"})

    # Find similar pairs
    pairs = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            # Cosine similarity
            dot = sum(a * b for a, b in zip(embeddings[i], embeddings[j]))
            norm_i = sum(a * a for a in embeddings[i]) ** 0.5
            norm_j = sum(a * a for a in embeddings[j]) ** 0.5
            if norm_i == 0 or norm_j == 0:
                continue
            sim = dot / (norm_i * norm_j)
            if sim >= threshold:
                pairs.append({
                    "memory_a": {"id": memories[i]["id"], "title": memories[i]["title"]},
                    "memory_b": {"id": memories[j]["id"], "title": memories[j]["title"]},
                    "similarity": round(sim, 3),
                    "action": "merge_into_a"  # By default, merge B into A
                })

    db.close()
    return json.dumps({
        "total_compared": len(memories),
        "duplicate_pairs": pairs,
        "threshold": threshold
    }, ensure_ascii=False)


def status_report():
    """Comparative report: runtime memory vs SQLite."""
    db = get_db()

    mem_count = db.execute("SELECT COUNT(*) FROM memories WHERE status='ativa'").fetchone()[0]
    chunk_count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
    categories = db.execute(
        "SELECT category, COUNT(*) FROM memories WHERE status='ativa' GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()

    db.close()

    report = {
        "sqlite": {
            "total_memories": mem_count,
            "total_chunks": chunk_count,
            "db_size_kb": round(db_size / 1024, 1),
            "model": MODEL_NAME,
            "categories": {r[0]: r[1] for r in categories},
        },
        "runtime_memory": {
            "max_chars": 2200,
            "note": "Check current usage via Hermes memory tool"
        },
        "capacity_ratio": f"~{mem_count * 500 // 2200}x more capacity in SQLite",
    }
    return json.dumps(report, ensure_ascii=False)


def export_obsidian(vault_path):
    """Export all SQLite memories to Obsidian notes."""
    db = get_db()
    vault = os.path.expanduser(vault_path)

    if not os.path.isdir(vault):
        os.makedirs(vault, exist_ok=True)

    memories = db.execute("SELECT * FROM memories WHERE status='ativa' ORDER BY category, title").fetchall()
    db.close()

    exported = 0
    for mem in memories:
        # Filename: safe title
        safe_title = re.sub(r'[^\w\s—-]', '', mem["title"]).strip()
        filename = f"{safe_title}.md"
        filepath = os.path.join(vault, filename)

        # Formatted content
        content = f"""# {mem['title']}

> Created: {mem['created_at'][:10]} | Last updated: {mem['updated_at'][:10]}
> Category: {mem['category']} | Tags: {mem['tags']}
> SQLite ID: {mem['id']}

{mem['content']}

## Links
- [[Labo — Memory Index]]
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        exported += 1

    # Create index
    index_path = os.path.join(vault, "Labo — Memory Index.md")
    index_content = "# Labo — Memory Index\n\n> Exported from SQLite on {}\n\n".format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    )

    # Group by category
    cats = {}
    for mem in memories:
        cat = mem["category"]
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(mem["title"])

    for cat, titles in sorted(cats.items()):
        index_content += f"\n## {cat.title()}\n\n"
        for title in titles:
            index_content += f"- [[{title}]]\n"

    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_content)

    return json.dumps({"exported": exported, "vault_path": vault, "index_created": True})


def main():
    parser = argparse.ArgumentParser(description="Labo LTM Manager — Hermes Integration")
    sub = parser.add_subparsers(dest="command")

    # offload
    p_off = sub.add_parser("offload", help="Offload runtime memory entries to SQLite")
    p_off.add_argument("entries", help="JSON array of entries to offload")

    # query
    p_q = sub.add_parser("query", help="Semantic search returning JSON context")
    p_q.add_argument("topic", help="Topic to search for")
    p_q.add_argument("--top_k", type=int, default=5, help="Max results")
    p_q.add_argument("--category", default=None, help="Filter by category")

    # context (alias for query with more compact output)
    p_ctx = sub.add_parser("context", help="Get relevant context as compact JSON")
    p_ctx.add_argument("topics", help="JSON array of topics to search")
    p_ctx.add_argument("--top_k", type=int, default=3)

    # init-session
    sub.add_parser("init-session", help="Get session startup context")

    # consolidate
    p_cons = sub.add_parser("consolidate", help="Find and propose merging duplicate memories")
    p_cons.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold")

    # status
    sub.add_parser("status", help="Runtime vs SQLite status report")

    # export-obsidian
    p_exp = sub.add_parser("export-obsidian", help="Export SQLite memories to Obsidian vault")
    p_exp.add_argument("path", help="Target vault path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "offload":
        print(offload_entries(args.entries))

    elif args.command == "query":
        print(query_context(args.topic, args.top_k, args.category))

    elif args.command == "context":
        topics = json.loads(args.topics) if isinstance(args.topics, str) else args.topics
        # Combine topics into one query
        combined = " ".join(topics) if isinstance(topics, list) else str(topics)
        print(query_context(combined, args.top_k))

    elif args.command == "init-session":
        print(init_session_context())

    elif args.command == "consolidate":
        print(consolidate_memories(args.threshold))

    elif args.command == "status":
        print(status_report())

    elif args.command == "export-obsidian":
        print(export_obsidian(args.path))


if __name__ == "__main__":
    main()
