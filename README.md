# Labo LTM MemoryProvider — Complete Installation Guide

> **For Hermes Agent users who want a permanent, local, semantic long-term memory system.**
>
> This guide is designed to be followed by **another Hermes Agent**. Each step includes
> verification checks so the installing agent can confirm success before proceeding.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Installation — Step by Step](#installation--step-by-step)
6. [Verification and Testing](#verification-and-testing)
7. [How It Works — Summary for the Installing Agent](#how-it-works--summary-for-the-installing-agent)
8. [Troubleshooting](#troubleshooting)
9. [Appendix: Full File Reference](#appendix-full-file-reference)

---

## Overview

The **LTM (Long-Term Memory) MemoryProvider** is a plugin for **Hermes Agent** that
replaces the built-in MEMORY.md / USER.md runtime memory with a **persistent semantic
memory system** powered by:

| Component | Role |
|---|---|
| **Granite-97m ONNX** (IBM) | Multilingual semantic embedding model (384-dim, int8 quantized) |
| **SQLite + sqlite-vec** | Vector database for similarity search |
| **Subprocess isolation** | LTM runs in its own Python venv — zero conflicts with Hermes deps |
| **Idle timeout** | Auto-exits after 15min idle (`LTM_IDLE_TIMEOUT`) — no orphaned processes |
| **Auto-restart** | Plugin auto-restarts subprocess if it exits (idle timeout or crash) |
| **MemoryProvider ABC** | 6 automatic hooks + 2 agent tool calls for full lifecycle |

### What it gives you

| Capability | Without LTM | With LTM |
|---|---|---|
| **Permanent storage** | MEMORY.md / USER.md (~2K chars each) | Unlimited SQLite |
| **Semantic search** | Manual grep / session_search | `ltm_search` tool — Granite ONNX vector search |
| **Automatic recall** | None | `prefetch()` — relevant context auto-injected per turn |
| **Persist decisions** | Manual memory tool | `ltm_add` tool — permanent with embedding |
| **Pre-compression save** | Lost on context compact | `on_pre_compress` — saves + prompts compressor LLM |
| **End-of-session save** | Lost on `/new` | `on_session_end` — saves + auto-export Obsidian |
| **Memory mirror** | Separate manual sync | `on_memory_write` — auto-mirrors runtime writes to LTM |

---

## Architecture

```
  HERMES AGENT (Python 3.x)               LTM SUBPROCESS (ltm-env, Python 3.x)
  ┌─────────────────────────┐             ┌──────────────────────────────┐
  │    LTMMemoryProvider     │  ────►     │        ltm_ops.py            │
  │                         │  stdin/     │  ┌──────── ─────────┐       │
  │  initialize()           │  stdout     │  │ search (Granite  │       │
  │  queue_prefetch()───►───┼───JSON───►──┤  │  ONNX + sqlite-) │       │
  │  prefetch() (cache)     │  ◄──────────┤  │  vec)            │       │
  │  sync_turn() (buffer)   │  response   │  ├──────────────────┤       │
  │  on_pre_compress()      │             │  │ add (embed +     │       │
  │  on_session_end()       │             │  │  insert)         │       │
  │  on_memory_write()      │             │  ├──────────────────┤       │
  │  on_delegation()        │             │  │ stats / ping     │       │
  │                         │             │  └──────────────────┘       │
  │  Tools: ltm_search      │             │  Depends on:                │
  │         ltm_add         │             │  query_memory.py            │
  └─────────────────────────┘             │  ltm_manager.py             │
                                          │  granite_embedder.py        │
                                          └───────────┬──────────────────┘
                                                      │
                                                      ▼
                                          ┌──────────────────────┐
                                          │  longterm-memory.db  │
                                          │  (SQLite + vec0)     │
                                          └──────────────────────┘
```

### Data flow per turn

```
USER sends message
  → MemoryManager calls queue_prefetch(user_message)
      → daemon thread: subprocess search → cache result
  → LLM processes + responds
  → MemoryManager calls sync_turn(user_msg, asst_msg)
      → buffer in RAM (no subprocess)
  → MemoryManager calls prefetch(next_user_message)
      → return cached result (instant)
```

### On context compression

```
MemoryManager calls on_pre_compress(messages_about_to_be_discarded)
  → async: subprocess add (save conversation to LTM)
  → return: prompt text for compression LLM
      "LLM, preserve important information from these messages"
```

### On session end (`/new` or disconnect)

```
MemoryManager calls on_session_end(remaining_messages)
  → flush turn buffer + save remaining conversation to LTM
  → if dirty: auto-export to Obsidian
MemoryManager calls on_session_switch(new_session_id, reset=True)
  → clear prefetch cache, clear turn buffer
```

---

## Repository Structure

This repository ships everything you need:

```
ltm-memory-provider/
├── README.md                       ← this guide
├── src/
│   ├── granite_embedder.py         ONNX model wrapper (68 lines)
│   ├── query_memory.py             SQLite DB + embedding pipeline (646 lines)
│   ├── ltm_manager.py              High-level search/export/status (526 lines)
│   └── ltm_ops.py                  Subprocess bridge (260 lines)
└── plugins/
    └── ltm/
        ├── plugin.yaml             Plugin configuration
        └── __init__.py             MemoryProvider implementation (706 lines)
```

**For the installing agent:** For each file, use:
1. `read_file("<path-in-this-repo>")` to read the content
2. `write_file("<destination-path>", <content>)` to create it

Full file contents are also in the [Appendix](#appendix-full-file-reference).

---

## Prerequisites

Before starting, **verify these are present** on the target system:

### Python 3.10+

```bash
python3 --version
```

Expected: `Python 3.10.x` or higher.

### Hermes Agent installed

```bash
hermes --version 2>/dev/null || python3 -m hermes_cli --version 2>/dev/null
```

Expected: version string.

### Hermes config exists

```bash
ls ~/.hermes/config.yaml
```

Expected: file exists.

### Storage space

The Granite ONNX model is ~395 MB. Database will be 1-5 MB.
Ensure at least **500 MB free** for the model + dependencies.

### Internet access (for model download)

The Granite ONNX model is downloaded automatically from HuggingFace Hub on first
inference. Subsequent runs use the local cache (~/.cache/huggingface/).

---

## Installation — Step by Step

---

### Step 1: Create Directories

```bash
mkdir -p ~/.hermes/models/granite-embedding-97m-multilingual-r2
mkdir -p ~/.hermes/ltm
mkdir -p ~/.hermes/plugins/ltm
```

**Verification:**
```bash
ls -d ~/.hermes/models/granite-embedding-97m-multilingual-r2 && \
ls -d ~/.hermes/ltm && \
ls -d ~/.hermes/plugins/ltm
```

Expected: all three directories exist.

---

### Step 2: Create and Activate the LTM Virtual Environment

The LTM runs in an isolated Python venv to avoid dependency conflicts.

```bash
# Create the venv
python3 -m venv ~/.hermes/ltm-env

# Activate and install dependencies
source ~/.hermes/ltm-env/bin/activate
pip install --upgrade pip
```

**Install the required packages:**

```bash
# Core ML dependencies
pip install numpy==1.26.4
pip install onnxruntime==1.18.1
pip install transformers==4.44.2
pip install tokenizers==0.19.1
pip install huggingface-hub==0.24.6

# Vector search
pip install sqlite-vec==0.1.6
```

> ❗ If pip fails, try adding `--timeout 120` to each command. The onnxruntime
> wheel is large (~30MB).
>
> ❗ If `python3 -m venv` is unavailable, install it via your system package
> manager (e.g. `sudo apt install python3-venv` on Debian/Ubuntu).

**Verification:**
```bash
python -c "
import numpy; print(f'numpy {numpy.__version__} ✅')
import onnxruntime; print(f'onnxruntime {onnxruntime.__version__} ✅')
import transformers; print(f'transformers {transformers.__version__} ✅')
import sqlite_vec; print(f'sqlite-vec ✅')
" 2>&1 | grep -v WARNING
```

Expected:
```
numpy 1.26.4 ✅
onnxruntime 1.18.1 ✅
transformers 4.44.2 ✅
sqlite-vec ✅
```

Deactivate when done:
```bash
deactivate
```

---

### Step 3: Create the LTM Core Scripts

This step creates the three core files the subprocess depends on.
For each file, the installing agent should read from this repository
and write to the destination.

| Source (in this repo) | Destination |
|---|---|
| `src/granite_embedder.py` | `~/.hermes/ltm/granite_embedder.py` |
| `src/query_memory.py` | `~/.hermes/ltm/query_memory.py` |
| `src/ltm_manager.py` | `~/.hermes/ltm/ltm_manager.py` |

**For the installing agent:**
```python
# Example workflow for each file:
content = read_file("/path/to/repo/src/granite_embedder.py")
write_file("~/.hermes/ltm/granite_embedder.py", content)
```

**Verification — test imports:**
```bash
~/.hermes/ltm-env/bin/python -c "
import sys; sys.path.insert(0, '$HOME/.hermes/ltm')
from granite_embedder import GraniteONNXEmbedder
print('  1. granite_embedder ✅')
from query_memory import get_db
print('  2. query_memory ✅')
from ltm_manager import query_context, status_report
print('  3. ltm_manager ✅')
"
```

Expected:
```
  1. granite_embedder ✅
  2. query_memory ✅
  3. ltm_manager ✅
```

---

### Step 4: Initialize the Database

```bash
~/.hermes/ltm-env/bin/python -c "
import sys; sys.path.insert(0, '$HOME/.hermes/ltm')
from query_memory import get_db, init_db, add_memory
db = get_db()
init_db(db)
import datetime
text = 'LTM MemoryProvider installed on ' + datetime.datetime.now().strftime('%Y-%m-%d')
add_memory(db, 'LTM System — Initialized', text, category='meta', tags='system', source='setup')
db.close()
print('Database initialized ✅')
"
```

**Verification:**
```bash
ls -la ~/.hermes/longterm-memory.db
```

Expected: file exists (100KB+).

---

### Step 5: Create the Subprocess Wrapper

| Source (in this repo) | Destination |
|---|---|
| `src/ltm_ops.py` | `~/.hermes/ltm/ltm_ops.py` |

**For the installing agent:** Same workflow as Step 3 — read from `src/ltm_ops.py`,
write to `~/.hermes/ltm/ltm_ops.py`.

**Verification — test the subprocess protocol:**

```bash
echo '{"cmd": "ping"}' | ~/.hermes/ltm-env/bin/python ~/.hermes/ltm/ltm_ops.py 2>/dev/null
```

Expected:
```json
{"ready": true}
{"pong": true, "model_loaded": true}
{"shutdown": true}
```

> ⚠️ First run downloads the Granite ONNX model (~395MB). Takes 2-5 minutes.

**Test search:**
```bash
printf '{"cmd": "search", "query": "LTM System initialization", "top_k": 3}\n' | \
  ~/.hermes/ltm-env/bin/python ~/.hermes/ltm/ltm_ops.py 2>/dev/null
```

Expected: JSON with `results_count: 1` (the welcome memory).

**Test stats:**
```bash
printf '{"cmd": "stats"}\n' | ~/.hermes/ltm-env/bin/python ~/.hermes/ltm/ltm_ops.py 2>/dev/null
```

Expected: JSON with `total_memories: 1`.

**Test error handling:**
```bash
printf '{"cmd": "unknown"}\n' | ~/.hermes/ltm-env/bin/python ~/.hermes/ltm/ltm_ops.py 2>/dev/null
```

Expected:
```json
{"ready": true}
{"error": "unknown command: unknown", "cmd": "unknown"}
{"shutdown": true}
```

If ALL pass, `ltm_ops.py` is working correctly. ✅

---

### Step 6: Create the Plugin Files

#### 6a. `plugin.yaml`

Write the content below to `~/.hermes/plugins/ltm/plugin.yaml`:

```yaml
name: ltm
version: 1.0.0
description: >
  Labo LTM MemoryProvider — Granite-97m ONNX embeddings + SQLite-vec.
  Subprocess-isolated (runs in ltm-env venv), fully local, zero external
  dependencies. Provides automatic recall via prefetch, semantic search via
  ltm_search tool, fact extraction on_pre_compress and on_session_end, and
  mirrors built-in memory writes to the LTM SQLite database.
hooks:
  - on_session_end
  - on_pre_compress
  - on_memory_write
  - on_session_switch
```

**Verification:**
```bash
cat ~/.hermes/plugins/ltm/plugin.yaml
```

#### 6b. `__init__.py` — the MemoryProvider

| Source (in this repo) | Destination |
|---|---|
| `plugins/ltm/__init__.py` | `~/.hermes/plugins/ltm/__init__.py` |

**For the installing agent:** Same workflow — read from `plugins/ltm/__init__.py`,
write to `~/.hermes/plugins/ltm/__init__.py`.

**Verification:**
```bash
wc -l ~/.hermes/plugins/ltm/__init__.py
```

Expected: ~737 lines.

---

### Step 7: Configure Hermes Agent

#### 7a. Set the provider

```bash
hermes config set memory.provider ltm
```

Expected:
```
✓ Set memory.provider = ltm in ~/.hermes/config.yaml
```

If `hermes config set` is unavailable, edit `~/.hermes/config.yaml` directly:
```yaml
memory:
  provider: ltm
```

#### 7b. Set plugin configuration (optional)

```bash
hermes config set plugins.ltm.search_top_k 3
```

If using Obsidian:
```bash
hermes config set plugins.ltm.obsidian_vault "~/Documents/Obsidian Vault"
hermes config set plugins.ltm.auto_export_obsidian true
```

##### Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `LTM_DB_PATH` | `~/.hermes/longterm-memory.db` | Custom path for the SQLite database. Useful for testing or per-project databases. |
| `LTM_IDLE_TIMEOUT` | `900` (15 min) | Seconds of inactivity before the subprocess auto-exits to free memory. Set to `0` to disable. |

#### 7c. Disable conflicting plugins

```bash
# Check for existing memory or LTM plugins
ls ~/.hermes/plugins/ 2>/dev/null

# If an old ltm_integration or similar plugin exists, rename it:
mv ~/.hermes/plugins/ltm_integration ~/.hermes/plugins/_ltm_integration_disabled 2>/dev/null
```

> ⚠️ Two plugins injecting LTM context will cause duplicate content.

---

### Step 8: Remove the Welcome Memory

Remove the test memory so the system starts clean:

```bash
~/.hermes/ltm-env/bin/python -c "
import sys; sys.path.insert(0, '$HOME/.hermes/ltm')
from query_memory import get_db, delete_memory
db = get_db()
delete_memory(db, 1)
db.close()
print('Welcome memory removed ✅')
"
```

**Verification:**
```bash
printf '{"cmd": "stats"}\n' | ~/.hermes/ltm-env/bin/python ~/.hermes/ltm/ltm_ops.py 2>/dev/null | \
  grep -o '"total_memories": [0-9]*'
```

Expected: `"total_memories": 0`

---

## Verification and Testing

The installing agent should execute each test and report results.

### Test 1: Plugin Discovery

```bash
cd ~/.hermes/hermes-agent && source venv/bin/activate && python3 -c "
from plugins.memory import discover_memory_providers
providers = discover_memory_providers()
ltm_found = any(n == 'ltm' for n, _, _ in providers)
print(f'LTM discovered: {ltm_found}')
for n, d, a in providers:
    if n == 'ltm':
        print(f'  {\"✅\" if a else \"❌\"} {n}: {d[:60]}')
"
```

Expected:
```
LTM discovered: True
  ✅ ltm: Labo LTM MemoryProvider — Granite-97m ONNX embeddings...
```

### Test 2: Provider Load

```bash
cd ~/.hermes/hermes-agent && source venv/bin/activate && python3 -c "
from plugins.memory import load_memory_provider
p = load_memory_provider('ltm')
print(f'Provider name: {p.name}')
print(f'Available: {p.is_available()}')
tools = [s['name'] for s in p.get_tool_schemas()]
print(f'Tools: {tools}')
print(f'Config entries: {len(p.get_config_schema())}')
"
```

Expected:
```
Provider name: ltm
Available: True
Tools: ['ltm_search', 'ltm_add']
Config entries: 3
```

### Test 3: Provider Initialization

```bash
cd ~/.hermes/hermes-agent && source venv/bin/activate && python3 << 'INITT'
import time
from plugins.memory import load_memory_provider
p = load_memory_provider("ltm")
t0 = time.time()
p.initialize(session_id="test-install")
dt = time.time() - t0
print(f"Initialize: {dt:.1f}s")
print(f"Initialized: {p._initialized}")
print("System prompt:")
print(p.system_prompt_block())
alive = p.check_alive()
print(f"Subprocess alive: {alive}")
p.shutdown()
print("Shutdown: OK")
INITT
```

Expected:
```
Initialize: 10-30s (first run: 2-5min model download)
Initialized: True
System prompt: # LTM (Long-Term Memory) ...
Subprocess alive: True
Shutdown: OK
```

### Test 4: End-to-End Session Simulation

```bash
cd ~/.hermes/hermes-agent && source venv/bin/activate && python3 << 'E2E'
import json, time
from plugins.memory import load_memory_provider

p = load_memory_provider("ltm")
p.initialize(session_id="install-test")

results = []

# 4a. ltm_add
r = p.handle_tool_call("ltm_add", {
    "title": "Installation Test Entry",
    "content": "This verifies the LTM MemoryProvider works correctly.",
    "category": "meta", "tags": "install-test",
})
data = json.loads(r)
ok = data.get("status") == "added"
results.append(("ltm_add", "✅" if ok else "❌"))
tid = data.get("memory_id")

# 4b. ltm_search
r = p.handle_tool_call("ltm_search", {"query": "installation test LTM", "top_k": 3})
data = json.loads(r)
ok = data.get("results_count", 0) >= 1
results.append(("ltm_search", "✅" if ok else "❌"))

# 4c. queue_prefetch + prefetch
p.queue_prefetch("LTM installation test", session_id="test")
time.sleep(5)
cached = p.prefetch("LTM installation test", session_id="test")
results.append(("prefetch", "✅" if len(cached) > 0 else "⚠️"))

# 4d. sync_turn
p.sync_turn("Test message", "Test response", session_id="test")
results.append(("sync_turn", "✅" if p._turn_count == 1 else "❌"))

# 4e. on_pre_compress
r = p.on_pre_compress([{"role": "user", "content": "Test decision for compression"}])
results.append(("on_pre_compress", "✅" if len(r) > 0 else "❌"))

# 4f. on_memory_write
p.on_memory_write("add", "user", "Test LTM memory mirror")
results.append(("on_memory_write", "✅"))

# 4g. on_session_switch
p.on_session_switch("new-session", reset=True)
results.append(("on_session_switch", "✅"))

# Summary
all_pass = all("❌" not in s for _, s in results)
print("TEST RESULTS:")
for name, status in results:
    print(f"  {status} {name}")
print(f"\nAll passed: {all_pass}")
print(f"Test entry ID to remove: {tid}")
p.shutdown()
E2E
```

Expected: All tests pass (no ❌).

### Test 5: Subprocess Stability

```bash
cd ~/.hermes/hermes-agent && source venv/bin/activate && python3 -c "
from plugins.memory import load_memory_provider
p = load_memory_provider('ltm')
p.initialize(session_id='stability-test')
for i in range(10):
    alive = p.check_alive()
    if not alive:
        print(f'FAIL at iteration {i+1}')
        break
print('10/10 pings successful ✅')
p.shutdown()
"
```

Expected: `10/10 pings successful ✅`

### Cleanup: Remove Test Entries

```bash
~/.hermes/ltm-env/bin/python -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/.hermes/ltm'))
from query_memory import get_db, delete_memory
db = get_db()
cursor = db.cursor()
cursor.execute('SELECT id FROM memories WHERE tags LIKE ?', ('%install-test%',))
for r in cursor.fetchall():
    delete_memory(db, r['id'])
cursor.execute('SELECT COUNT(*) FROM memories')
count = cursor.fetchone()[0]
print(f'Memories remaining: {count}')
db.close()
"
```

Expected: 0 memories remaining (clean start).

---

## How It Works — Summary for the Installing Agent

When installation completes, deliver this to the user:

---

> ✅ **LTM MemoryProvider installed and functional.**
>
> The Hermes Agent now has a permanent, local, semantic long-term memory system.
>
> **What was installed:**
> - **`~/.hermes/ltm/ltm_ops.py`** — Persistent subprocess that loads Granite-97m
>   ONNX and serves search/addition requests
> - **`~/.hermes/plugins/ltm/plugin.yaml`** — Plugin registration with hooks
> - **`~/.hermes/plugins/ltm/__init__.py`** — Full MemoryProvider (27 methods, 11 hooks + 2 tools)
> - **`~/.hermes/longterm-memory.db`** — SQLite database with sqlite-vec
> - **`~/.hermes/ltm-env/`** — Isolated Python venv
>
> **How it works in daily use:**
>
> | When this happens... | The LTM automatically... |
> |---|---|
> | Session starts | Loads the Granite ONNX model (~10s) |
> | User sends a message | Queues background semantic search (cached for next turn) |
> | Agent receives message | Injects cached context into system prompt (instant) |
> | Each turn | Buffers the exchange in RAM |
> | Context is compressed | Saves messages to LTM + prompts LLM to preserve key info |
> | Session ends (`/new`) | Saves remaining conversation + exports to Obsidian |
> | Agent writes to memory tool | Mirrors the write to LTM |
> | Subagent finishes task | Saves notable results to LTM |
>
> **Two new agent tools:**
> - **`ltm_search`** — Semantic vector search across ALL past conversations.
>   Always search before asking the user to repeat information.
> - **`ltm_add`** — Permanently store important decisions, rules, preferences.
>   Embedded with Granite ONNX and stored in SQLite.
>
> **Configuration:**
> - Provider: `memory.provider: ltm`
> - Plugin options: `plugins.ltm.*` in config.yaml
> - Auto-export to Obsidian: Yes (if configured)
> - Zero external API calls — everything runs locally.

---

## Troubleshooting

### Plugin not discovered

**Check:**
```bash
ls -la ~/.hermes/plugins/ltm/
# Must show: __init__.py  plugin.yaml
```

**Fix:** Ensure both files exist.

### Subprocess won't start

**Check:**
```bash
echo '{"cmd": "ping"}' | ~/.hermes/ltm-env/bin/python ~/.hermes/ltm/ltm_ops.py 2>&1
```

**Fix:** Verify venv packages, disk space, and Python syntax.

### Model fails to load

**Check:**
```bash
ls -la ~/.hermes/models/granite-embedding-97m-multilingual-r2/
```

**Fix:** Delete and retry:
```bash
rm -rf ~/.hermes/models/granite-embedding-97m-multilingual-r2
# Then re-run the subprocess test — it triggers a fresh download
```

### Provider shows "Available: False"

The `is_available()` method checks three files exist:
1. `~/.hermes/ltm-env/bin/python`
2. `~/.hermes/ltm/ltm_ops.py`
3. `~/.hermes/longterm-memory.db`

**Fix:** Create the missing file by re-running the appropriate step.

### Cannot import query_memory

This means the Python path isn't set. The subprocess adds the LTM directory
to `sys.path` automatically:
```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

**Verify:**
```bash
~/.hermes/ltm-env/bin/python -c "
import sys; sys.path.insert(0, '$HOME/.hermes/ltm')
from query_memory import get_db; print('OK ✅')
"
```

---

## Appendix: Full File Reference

This appendix contains the complete source code for all files. The installing
agent can either copy files from the repository directory or create them using
the content below.

### A1: `granite_embedder.py` (68 lines)

**Source:** `src/granite_embedder.py` in this repository.

```python
"""
Granite ONNX Embedder — Wrapper ONNX int8 para sentence-transformers compat.
Carrega granite-embedding-97m-multilingual-r2 via ONNX Runtime (int8 quantizado).
"""
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import os, warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_DIR = os.path.expanduser("~/.hermes/models/granite-embedding-97m-multilingual-r2")
ONNX_PATH = os.path.join(MODEL_DIR, "onnx", "model_quint8_avx2.onnx")
EMBEDDING_DIM = 384
MAX_LENGTH = 8192

class GraniteONNXEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="np", max_length=MAX_LENGTH)
        ort_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        if "token_type_ids" in self.input_names:
            ort_inputs["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))
        outputs = self.session.run(None, ort_inputs)
        token_embeddings = outputs[0]
        mask = np.expand_dims(inputs["attention_mask"], axis=-1).astype(token_embeddings.dtype)
        sum_embeddings = np.sum(token_embeddings * mask, axis=1)
        sum_mask = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask
        if normalize_embeddings:
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / norm
        return pooled.tolist()
```

### A2: `query_memory.py` (646 lines)

**Source:** `src/query_memory.py` in this repository.

Comprehensive file that provides:
- SQLite connection with WAL mode, sqlite-vec vector extension
- `get_db()`, `init_db(db)` — schema creation (`memories`, `chunks`, `vec_chunks` tables)
- `add_memory(db, title, content, category, tags, source)` — insert + embed
- `search_memory(db, query, top_k, category)` — vector similarity search
- `get_memory(db, id)`, `update_memory(...)`, `delete_memory(db, id)`
- `list_memories(db, category, status, limit)`
- `import_vault(db, path)` — import Obsidian .md files
- `reindex_all(db)` — regenerate all embeddings
- Granite ONNX model singleton (`_get_model()`)
- Text chunking (`chunk_text()`)
- Embedding conversion utilities (`embedding_to_blob()`, `vec_embedding()`)
- CLI parser for standalone use

### A3: `ltm_manager.py` (526 lines)

**Source:** `src/ltm_manager.py` in this repository.

Provides high-level operations:
- `classify_category(text)` — keyword-based categorization
- `offload_entries(entries_json)` — migrate runtime entries to SQLite
- `query_context(topic, top_k, category)` — semantic search returning JSON
- `init_session_context()` — session startup info
- `consolidate_memories(threshold)` — dedup similar entries
- `status_report()` — DB statistics as JSON
- `export_obsidian(vault_path)` — full Obsidian vault export
- CLI parser for standalone use

### A4: `ltm_ops.py` (260 lines)

**Source:** `src/ltm_ops.py` in this repository.

The persistent subprocess bridge — reads JSON commands from stdin, writes JSON
responses to stdout. Commands: `search`, `add`, `add_batch`, `stats`,
`init_session`, `export_obsidian`, `ping`. Imports from `query_memory`
and `ltm_manager` internally.

Features:
- **Idle timeout** — auto-exits after 15 minutes of inactivity (configurable
  via `LTM_IDLE_TIMEOUT` env var) to prevent orphaned subprocesses.
- **Model warm-up** — pre-loads Granite ONNX on startup so the first search
  is fast.

### A5: `plugin.yaml`

**Source:** `plugins/ltm/plugin.yaml` in this repository.

```yaml
name: ltm
version: 1.0.0
description: >
  Labo LTM MemoryProvider — Granite-97m ONNX embeddings + SQLite-vec.
  Subprocess-isolated (runs in ltm-env venv), fully local, zero external
  dependencies. Provides automatic recall via prefetch, semantic search via
  ltm_search tool, fact extraction on_pre_compress and on_session_end, and
  mirrors built-in memory writes to the LTM SQLite database.
  Features idle timeout (LTM_IDLE_TIMEOUT) and auto-restart for resilience.
hooks:
  - on_session_end
  - on_pre_compress
  - on_memory_write
  - on_session_switch
```

### A6: `__init__.py` (737 lines)

**Source:** `plugins/ltm/__init__.py` in this repository.

The MemoryProvider ABC implementation. Contains:
- **`LTMMemoryProvider` class** with 28 methods:
  - Core: `__init__`, `name`, `is_available`, `get_config_schema`, `save_config`
  - Lifecycle: `initialize`, `system_prompt_block`, `shutdown`, `check_alive`
  - Recall: `queue_prefetch`, `prefetch`, `_format_prefetch`
  - Persistence: `sync_turn`
  - Tools: `get_tool_schemas`, `handle_tool_call`, `_handle_ltm_search`, `_handle_ltm_add`
  - Hooks: `on_pre_compress`, `on_session_end`, `on_memory_write`, `on_session_switch`, `on_delegation`
  - Resilience: `_ensure_alive` — auto-restarts subprocess after idle timeout or crash
  - Internal: `_is_alive`, `_send`, `_send_async`, `_extract_excerpts`
- **`register(ctx)`** — plugin entry point
- **`_load_plugin_config()`** — config loader
- **`LTM_SEARCH_SCHEMA`**, **`LTM_ADD_SCHEMA`** — tool definitions

---

*LTM MemoryProvider for Hermes Agent*
