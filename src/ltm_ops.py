#!/usr/bin/env python3
"""
LTM Operations — Persistent subprocess for Hermes Agent MemoryProvider plugin.

Line-based JSON protocol: reads one JSON command per line from stdin,
writes one JSON response per line to stdout. Designed to run as a long-lived
process spawned by the LTM MemoryProvider plugin, keeping the ONNX model
loaded in memory across multiple operations.

Runs in LTM venv Python (~/.hermes/ltm-env/bin/python) with access to
onnxruntime, sqlite-vec, transformers, and the Granite-97m ONNX model.

Commands:
  {"cmd": "search", "query": "...", "top_k": 5, "category": null}
  {"cmd": "add", "title": "...", "content": "...", "category": "general", "tags": "", "source": "labo"}
  {"cmd": "add_batch", "entries": [{"title": "...", "content": "...", "category": "...", ...}]}
  {"cmd": "stats"}
  {"cmd": "init_session"}
  {"cmd": "export_obsidian", "path": "..."}
  {"cmd": "ping"}

Responses are always single-line JSON. Errors are caught and returned as:
  {"error": "message", "cmd": "search"}
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import sys
import io
import json
import logging
import time
import select
from contextlib import redirect_stdout, redirect_stderr

# Add LTM directory to path so we can import query_memory and ltm_manager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import LTM infrastructure
from query_memory import (
    get_db, init_db, add_memory, DB_PATH, _get_model,
    embed_texts, vec_embedding,
)
from ltm_manager import (
    query_context, status_report, init_session_context,
    export_obsidian, classify_category,
)

logger = logging.getLogger("ltm_ops")
logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")


def _safe_call(func, *args, **kwargs):
    """Call func, suppressing stdout/stderr (e.g. print() in add_memory)."""
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    with redirect_stdout(captured_out), redirect_stderr(captured_err):
        result = func(*args, **kwargs)
    return result


def handle_search(data):
    """Semantic search using Granite ONNX embeddings."""
    query = data.get("query", "")
    top_k = int(data.get("top_k", 5))
    category = data.get("category")

    if not query or len(query.strip()) < 2:
        return {"error": "query too short", "cmd": "search"}

    # query_context returns a JSON string — parse it back
    raw = _safe_call(query_context, query, top_k, category)
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def handle_add(data):
    """Add a single memory with Granite embedding."""
    title = data.get("title", "")
    content = data.get("content", "")
    category = data.get("category", "general")
    tags = data.get("tags", "")
    source = data.get("source", "labo")

    if not title or not content:
        return {"error": "title and content required", "cmd": "add"}
    if len(content) < 10:
        return {"error": "content too short", "cmd": "add"}

    db = get_db()
    try:
        mem_id = _safe_call(add_memory, db, title, content, category, tags, source)
        db.close()
        return {"memory_id": mem_id, "status": "added", "title": title, "category": category}
    except Exception as e:
        db.close()
        return {"error": str(e), "cmd": "add"}


def handle_add_batch(data):
    """Add multiple memories in one call (for on_pre_compress / on_session_end)."""
    entries = data.get("entries", [])
    if not entries:
        return {"added": 0, "errors": [], "cmd": "add_batch"}

    db = get_db()
    added = []
    errors = []

    for entry in entries:
        title = entry.get("title", "")
        content = entry.get("content", "")
        category = entry.get("category") or classify_category(f"{title} {content}")
        tags = entry.get("tags", "auto-extracted")
        source = entry.get("source", "ltm-provider")

        if not title or not content or len(content) < 10:
            errors.append({"title": title, "reason": "invalid"})
            continue

        try:
            mem_id = _safe_call(add_memory, db, title, content, category, tags, source)
            added.append({"memory_id": mem_id, "title": title, "category": category})
        except Exception as e:
            errors.append({"title": title, "error": str(e)})

    db.close()
    return {"added": len(added), "entries": added, "errors": errors, "cmd": "add_batch"}


def handle_stats(data):
    """Return DB statistics."""
    raw = _safe_call(status_report)
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def handle_init_session(data):
    """Return session startup context (stats + categories + recent)."""
    raw = _safe_call(init_session_context)
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def handle_export_obsidian(data):
    """Export all memories to Obsidian vault."""
    path = data.get("path", "")
    if not path:
        return {"error": "path required", "cmd": "export_obsidian"}
    raw = _safe_call(export_obsidian, path)
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def handle_ping(data):
    """Health check — confirm subprocess is alive and model is loaded."""
    return {"pong": True, "model_loaded": _get_model() is not None}


# Command dispatch
COMMANDS = {
    "search": handle_search,
    "add": handle_add,
    "add_batch": handle_add_batch,
    "stats": handle_stats,
    "init_session": handle_init_session,
    "export_obsidian": handle_export_obsidian,
    "ping": handle_ping,
}


def main():
    # Warm up the ONNX model at startup so first search is fast
    try:
        _get_model()
    except Exception as e:
        # If model fails to load, we still run — search will error gracefully
        sys.stderr.write(f"WARNING: Model warm-up failed: {e}\n")

    # Ensure DB schema exists (suppress init_db's print output)
    try:
        db = get_db()
        _safe_call(init_db, db)
        db.close()
    except Exception:
        pass  # Schema likely already exists

    # Signal readiness
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    # Idle timeout: auto-exit after this many seconds without commands
    # Prevents orphaned subprocesses when the parent agent doesn't shut down cleanly
    IDLE_TIMEOUT = int(os.environ.get("LTM_IDLE_TIMEOUT", "900"))  # default 15min
    _last_cmd_time = time.time()

    # Main loop: poll stdin with timeout so we can detect idle
    while True:
        # Check idle timeout
        idle = time.time() - _last_cmd_time
        if idle > IDLE_TIMEOUT:
            # No commands for too long — auto-exit to free memory
            break

        # Poll stdin with 5s granularity so idle check runs frequently
        readable, _, _ = select.select([sys.stdin], [], [], 5)
        if not readable:
            continue  # Nothing to read — loop back and check idle

        line = sys.stdin.readline()
        if not line:
            break  # EOF — parent closed pipe

        line = line.strip()
        if not line:
            continue

        # Reset idle timer on any command
        _last_cmd_time = time.time()

        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            response = {"error": f"invalid JSON: {e}", "cmd": "unknown"}
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue

        cmd = data.get("cmd", "")
        handler = COMMANDS.get(cmd)

        if handler is None:
            response = {"error": f"unknown command: {cmd}", "cmd": cmd}
        else:
            try:
                response = handler(data)
            except Exception as e:
                logger.exception("Error handling cmd=%s", cmd)
                response = {"error": str(e), "cmd": cmd}

        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    # stdin closed — graceful shutdown
    sys.stdout.write(json.dumps({"shutdown": True}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
