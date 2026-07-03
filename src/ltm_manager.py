#!/usr/bin/env python3
"""
Labo Long-Term Memory Manager — Integração Hermes ↔ SQLite + Granite-97m

Camada de alto nível que o Hermes Agent usa para:
  1. OFFLOAD: migrar entradas da runtime memory p/ SQLite (libera espaço)
  2. CONSULTA: busca semântica automática por tema da sessão
  3. CONSOLIDAÇÃO: dedup + merge de memórias similares
  4. SINCRONIZAÇÃO: manter runtime memory enxuta com ponteiros

Uso:
  ltm_manager.py offload <json_entries>     # Recebe JSON da runtime, arquiva no SQLite
  ltm_manager.py query <topic> [--top_k 5]  # Busca semântica por tema
  ltm_manager.py context <topics_json>       # Gera contexto relevante para injeção no prompt
  ltm_manager.py consolidate [--threshold 0.85]  # Dedup de memórias similares
  ltm_manager.py status                      # Relatório: runtime vs SQLite
  ltm_manager.py export-obsidian <path>      # Exporta tudo para formato Obsidian
  ltm_manager.py init-session                # Retorna contexto relevante para nova sessão
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import datetime

# Reutilizar infra do query_memory
sys.path.insert(0, os.path.dirname(__file__))
from query_memory import (
    DB_PATH, VENV_PYTHON, MODEL_NAME, EMBEDDING_DIM,
    get_db, init_db, embed_texts, chunk_text,
    embedding_to_blob, vec_embedding, iso_now,
    add_memory, search_memory as _search_memory,
    get_memory, update_memory, delete_memory, list_memories,
    stats_db
)

# Categorias mapeadas por tipo de conteúdo
CATEGORY_KEYWORDS = {
    "infraestrutura": ["infra", "servidor", "cronjob", "deploy", "docker", "nginx", "ssh", "firewall", "rede", "vpn"],
    "config": ["config", "yaml", ".env", "variável", "variavel", "setting", "instalação", "instalacao", "setup"],
    "projeto": ["projeto", "bot", "trading", "pipeline", "app", "sistema", "feature", "módulo", "modulo"],
    "pesquisa": ["pesquisa", "framework", "biblioteca", "api", "sdk", "referência", "comparação", "benchmark"],
    "decisao": ["decisão", "decisao", "debate", "trade-off", "escolha", "arquitetura", "approach"],
    "correcao": ["nunca", "sempre", "não fazer", "nao fazer", "obrigatório", "obrigatorio", "regra", "prefere"],
    "debug": ["erro", "bug", "fix", "workaround", "falha", "timeout", "crash", "debug"],
}

# Entradas que NÃO devem ser offloaded (ficam na runtime)
RUNTIME_ONLY_PATTERNS = [
    r"chama.*Labo",
    r"respostas concisas",
    r"NUNCA instalar",
    r"NUNCA criar cron",
    r"modelo padrão",
    r"StealthyFetcher",
    r"prefere",
    r"Navarro",
]


def classify_category(text):
    """Classifica o texto em uma categoria baseado em keywords."""
    text_lower = text.lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[cat] = score
    if scores:
        return max(scores, key=scores.get)
    return "geral"


def should_stay_runtime(text):
    """Verifica se a entrada deve permanecer na runtime memory."""
    for pattern in RUNTIME_ONLY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def offload_entries(entries_json):
    """
    Recebe JSON com entradas da runtime memory e arquiva no SQLite.
    Retorna JSON com: offloaded (sucesso), skipped (ficam na runtime), errors.
    
    Formato de entrada: [{"text": "...", "target": "memory"|"user"}, ...]
    """
    try:
        entries = json.loads(entries_json) if isinstance(entries_json, str) else entries_json
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"JSON inválido: {e}"})

    db = get_db()
    # Garantir schema
    try:
        init_db(db)
    except Exception:
        pass  # Schema já existe

    result = {"offloaded": [], "skipped": [], "errors": []}

    for entry in entries:
        text = entry.get("text", "").strip()
        target = entry.get("target", "memory")

        if not text or len(text) < 20:
            result["skipped"].append({"reason": "muito_curto", "text": text[:80]})
            continue

        # Verificar se deve ficar na runtime
        if should_stay_runtime(text):
            result["skipped"].append({"reason": "runtime_essential", "text": text[:80]})
            continue

        # Classificar categoria
        category = classify_category(text)

        # Gerar título automático
        title = generate_title(text, target)

        # Verificar dedup por similaridade (busca antes de inserir)
        try:
            dup_check = check_duplicate(db, text, threshold=0.88)
            if dup_check:
                result["skipped"].append({
                    "reason": "duplicata",
                    "existing_id": dup_check,
                    "text": text[:80]
                })
                continue
        except Exception:
            pass  # Se busca falhar, prosseguir com inserção

        # Inserir no SQLite
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
    """Verifica se já existe memória similar. Retorna ID se duplicata, None caso contrário."""
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
            # Converter distância euclidiana para similaridade de cosseno
            similarity = 1 - (results[0]["distance"] ** 2) / 2
            if similarity >= threshold:
                return results[0]["memory_id"]
    except Exception:
        pass
    return None


def generate_title(text, target="memory"):
    """Gera título automático para a entrada."""
    # Pegar primeira linha significativa
    lines = text.strip().split("\n")
    first_line = ""
    for line in lines:
        clean = line.strip().lstrip("•-*§→").strip()
        if clean and len(clean) > 5:
            first_line = clean
            break
    
    if not first_line:
        first_line = text[:60]
    
    # Truncar
    title = first_line[:80]
    if len(first_line) > 80:
        title = title.rsplit(" ", 1)[0] + "..."
    
    # Prefixo por target
    prefix = "User" if target == "user" else "Labo"
    return f"{prefix} — {title}"


def query_context(topic, top_k=5, category=None):
    """
    Busca semântica e retorna JSON com contexto relevante.
    Usado pelo Hermes para injetar contexto no prompt.
    """
    db = get_db()

    try:
        query_emb = embed_texts([topic])[0]
    except Exception as e:
        db.close()
        return json.dumps({"error": f"Erro gerando embedding: {e}"})

    k = top_k * 3  # Buscar mais para deduplicar
    sql = """
    SELECT
        vec.distance,
        c.memory_id,
        c.chunk_text,
        m.title,
        m.category,
        m.tags,
        m.content,
        m.updated_at
    FROM vec_chunks vec
    JOIN chunks c ON vec.chunk_id = c.id
    JOIN memories m ON c.memory_id = m.id
    WHERE m.status = 'ativa'
    AND vec.embedding MATCH ?
    AND k = ?
    """
    params = [vec_embedding(query_emb), k]

    if category:
        sql += " AND m.category = ?"
        params.append(category)

    sql += " ORDER BY vec.distance ASC"

    try:
        results = db.execute(sql, params).fetchall()
    except Exception as e:
        db.close()
        return json.dumps({"error": f"Erro na busca: {e}"})

    # Deduplicar por memory_id
    seen = set()
    context_entries = []
    for r in results:
        mid = r["memory_id"]
        if mid in seen:
            continue
        seen.add(mid)

        # Converter distância euclidiana para similaridade de cosseno
        # Para vetores normalizados: cosine_sim = 1 - distance² / 2
        similarity = 1 - (r["distance"] ** 2) / 2
        if similarity < 0.30:  # Threshold ajustado para similaridade de cosseno
            continue

        context_entries.append({
            "id": mid,
            "title": r["title"],
            "category": r["category"],
            "tags": r["tags"],
            "similarity": round(similarity, 3),
            "updated": r["updated_at"],
            "relevant_chunk": r["chunk_text"][:500],
            "full_content": r["content"] if similarity > 0.45 else None,
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
    """
    Retorna contexto resumido para início de sessão.
    Combina: stats do DB + categorias com mais memórias + últimas atualizações.
    """
    db = get_db()

    # Stats gerais
    mem_count = db.execute("SELECT COUNT(*) FROM memories WHERE status='ativa'").fetchone()[0]
    
    # Categorias
    cats = db.execute(
        "SELECT category, COUNT(*) as c FROM memories WHERE status='ativa' GROUP BY category ORDER BY c DESC"
    ).fetchall()
    
    # Últimas 5 atualizações
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
    """
    Busca memórias similares e propõe merge.
    Retorna JSON com pares duplicados para aprovação.
    """
    db = get_db()
    memories = db.execute(
        "SELECT id, title, content, category FROM memories WHERE status='ativa' ORDER BY id"
    ).fetchall()

    if len(memories) < 2:
        db.close()
        return json.dumps({"message": "Poucas memórias para consolidar", "pairs": []})

    # Comparar todos os pares via embeddings
    # Para performance: embed tudo de uma vez
    texts = [f"{m['title']} {m['content'][:200]}" for m in memories]
    
    try:
        embeddings = embed_texts(texts)
    except Exception as e:
        db.close()
        return json.dumps({"error": f"Erro gerando embeddings: {e}"})

    # Encontrar pares similares
    pairs = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            # Similaridade de cosseno
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
                    "action": "merge_into_a"  # Por default, merge B → A
                })

    db.close()
    return json.dumps({
        "total_compared": len(memories),
        "duplicate_pairs": pairs,
        "threshold": threshold
    }, ensure_ascii=False)


def status_report():
    """Relatório comparativo runtime vs SQLite."""
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
            "note": "Verificar uso atual via memory tool do Hermes"
        },
        "capacity_ratio": f"~{mem_count * 500 // 2200}x mais capacidade no SQLite",
    }
    return json.dumps(report, ensure_ascii=False)


def export_obsidian(vault_path):
    """Exporta todas as memórias do SQLite para notas Obsidian."""
    db = get_db()
    vault = os.path.expanduser(vault_path)
    
    if not os.path.isdir(vault):
        os.makedirs(vault, exist_ok=True)
    
    memories = db.execute("SELECT * FROM memories WHERE status='ativa' ORDER BY category, title").fetchall()
    db.close()
    
    exported = 0
    for mem in memories:
        # Nome do arquivo: categoria + título
        safe_title = re.sub(r'[^\w\s—-]', '', mem["title"]).strip()
        filename = f"{safe_title}.md"
        filepath = os.path.join(vault, filename)
        
        # Conteúdo formatado
        content = f"""# {mem['title']}

> Criado em: {mem['created_at'][:10]} | Última atualização: {mem['updated_at'][:10]}
> Categoria: {mem['category']} | Tags: {mem['tags']}
> ID SQLite: {mem['id']}

{mem['content']}

## Links
- [[Labo — Memória Index]]
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        exported += 1
    
    # Criar index
    index_path = os.path.join(vault, "Labo — Memória Index.md")
    index_content = "# Labo — Memória Index\n\n> Exportado do SQLite em {}\n\n".format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    
    # Agrupar por categoria
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
    parser = argparse.ArgumentParser(description="Labo LTM Manager — Integração Hermes")
    sub = parser.add_subparsers(dest="command")

    # offload
    p_off = sub.add_parser("offload", help="Offload runtime memory entries to SQLite")
    p_off.add_argument("entries", help="JSON array of entries to offload")

    # query
    p_q = sub.add_parser("query", help="Semantic search returning JSON context")
    p_q.add_argument("topic", help="Topic to search for")
    p_q.add_argument("--top_k", type=int, default=5, help="Max results")
    p_q.add_argument("--category", default=None, help="Filter by category")

    # context (alias para query com output mais compacto)
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
        # Combinar tópicos em uma query
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
