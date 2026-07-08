"""
Labo LTM MemoryProvider — Granite ONNX + SQLite-vec, subprocess-isolated.

Provides:
  - Automatic recall via prefetch/queue_prefetch (background Granite ONNX search)
  - Semantic search via ltm_search tool (explicit, on-demand)
  - Permanent storage via ltm_add tool (explicit, on-demand)
  - Pre-compression extraction via on_pre_compress (saves to LTM + returns prompt)
  - Session-end extraction via on_session_end (saves remaining conversation)
  - Runtime memory mirroring via on_memory_write (auto-backup to LTM)

Architecture:
  The plugin spawns a persistent subprocess running in the LTM venv
  (~/.hermes/ltm-env/bin/python ~/.hermes/ltm/ltm_ops.py) to keep the
  Granite-97m ONNX model loaded in memory across operations. Communication
  is line-based JSON: write one JSON line to stdin, read one JSON line
  from stdout. Thread-safe via a threading.Lock around pipe access.

Config in $HERMES_HOME/config.yaml:
  memory:
    provider: ltm
  plugins:
    ltm:
      obsidian_vault: "~/Documents/Obsidian Vault"
      auto_export_obsidian: true
      search_top_k: 3
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import datetime
import select
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LTM_VENV_PYTHON = os.path.expanduser("~/.hermes/ltm-env/bin/python")
LTM_OPS_SCRIPT = os.path.expanduser("~/.hermes/ltm/ltm_ops.py")
LTM_DB_PATH = os.path.expanduser("~/.hermes/longterm-memory.db")
DEFAULT_OBSIDIAN_VAULT = os.path.expanduser("~/Documents/Obsidian Vault")
DEFAULT_SEARCH_TOP_K = 3
SUBPROCESS_TIMEOUT = 30  # seconds for subprocess to respond
SUBPROCESS_READY_TIMEOUT = 60  # seconds for initial model load


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

LTM_SEARCH_SCHEMA = {
    "name": "ltm_search",
    "description": (
        "Search the LTM (Long-Term Memory) database using Granite ONNX semantic embeddings. "
        "Returns relevant memories with similarity scores. "
        "Use when you need to recall past decisions, preferences, project details, "
        "environment specs, or context from previous conversations that may not be "
        "in the current runtime memory.\n\n"
        "The LTM contains curated, high-quality information — every entry was "
        "intentionally stored. Always search before asking the user to repeat information."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for. Natural language (PT-BR or EN). "
                               "e.g. 'configuração do PC', 'regras do pipeline', "
                               "'preferências de backtest trading bot'",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results to return (default: 5).",
                "default": 5,
            },
            "category": {
                "type": "string",
                "description": "Filter by category: config, projeto, decisao, geral, "
                               "infraestrutura, pesquisa, meta, devops, debug, x-post",
            },
        },
        "required": ["query"],
    },
}

LTM_ADD_SCHEMA = {
    "name": "ltm_add",
    "description": (
        "Add a permanent memory to the LTM database. The memory will be embedded "
        "with Granite ONNX and stored in SQLite for permanent recall across sessions. "
        "Use for important information the user would expect you to remember: "
        "decisions, preferences, project specs, rules, environment details.\n\n"
        "Do NOT use for transient information (current task progress, temporary "
        "state, session-specific context). Use the regular memory tool for those."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short, descriptive title (max 80 chars).",
            },
            "content": {
                "type": "string",
                "description": "Full content to store. Be detailed — this is permanent.",
            },
            "category": {
                "type": "string",
                "description": "Category: config, projeto, decisao, geral, "
                               "infraestrutura, pesquisa, meta",
                "default": "geral",
            },
            "tags": {
                "type": "string",
                "description": "Comma-separated tags for organization.",
                "default": "",
            },
        },
        "required": ["title", "content"],
    },
}


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    """Load plugin config from $HERMES_HOME/config.yaml -> plugins.ltm."""
    try:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
        if not config_path.exists():
            return {}
        import yaml
        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return all_config.get("plugins", {}).get("ltm", {}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# LTM MemoryProvider
# ---------------------------------------------------------------------------

class LTMMemoryProvider:
    """Labo LTM MemoryProvider — Granite ONNX + SQLite-vec via subprocess."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._prefetch_cache: str = ""
        self._turn_count: int = 0
        self._turn_buffer: List[Dict[str, str]] = []
        self._total_memories: int = 0
        self._obsidian_dirty: bool = False  # True when new memories added -> needs export
        self._obsidian_vault: str = os.path.expanduser(
            self._config.get("obsidian_vault", DEFAULT_OBSIDIAN_VAULT)
        )
        self._auto_export: bool = self._config.get("auto_export_obsidian", True)
        self._search_top_k: int = int(self._config.get("search_top_k", DEFAULT_SEARCH_TOP_K))
        self._auto_recall: bool = self._config.get("auto_recall", True)
        self._initialized: bool = False

    # -- Properties ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "ltm"

    # -- Core lifecycle ------------------------------------------------------

    def is_available(self) -> bool:
        """Check if LTM venv, ops script, and DB exist."""
        return (
            os.path.isfile(LTM_VENV_PYTHON)
            and os.path.isfile(LTM_OPS_SCRIPT)
            and os.path.isfile(LTM_DB_PATH)
        )

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "obsidian_vault",
                "description": "Path to Obsidian vault for auto-export",
                "default": DEFAULT_OBSIDIAN_VAULT,
            },
            {
                "key": "auto_export_obsidian",
                "description": "Auto-export LTM to Obsidian on session end",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "search_top_k",
                "description": "Default number of results for prefetch recall",
                "default": str(DEFAULT_SEARCH_TOP_K),
            },
            {
                "key": "auto_recall",
                "description": "Auto-inject LTM context into every turn (system prompt block + prefetch recall)",
                "default": "true",
                "choices": ["true", "false"],
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write config to config.yaml under plugins.ltm."""
        try:
            from pathlib import Path
            config_path = Path(hermes_home) / "config.yaml"
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["ltm"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception as e:
            logger.warning("Failed to save LTM config: %s", e)

    def initialize(self, session_id: str, **kwargs) -> None:
        """Spawn the persistent LTM subprocess and warm up the Granite model."""
        agent_context = kwargs.get("agent_context", "primary")
        if agent_context != "primary":
            # Subagents and cron contexts should not have LTM provider
            logger.info("LTM: skipping init for agent_context=%s", agent_context)
            return

        try:
            self._proc = subprocess.Popen(
                [LTM_VENV_PYTHON, LTM_OPS_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Wait for "ready" signal (model warm-up can take ~7.5s on i3-3240)
            ready_line = self._proc.stdout.readline()
            if not ready_line:
                raise RuntimeError("LTM subprocess closed before ready signal")
            ready_data = json.loads(ready_line)
            if not ready_data.get("ready"):
                raise RuntimeError(f"LTM subprocess not ready: {ready_data}")

            logger.info("LTM subprocess ready (model loaded)")

            # Get initial stats for system_prompt_block
            stats = self._send({"cmd": "stats"})
            if stats and "sqlite" in stats:
                self._total_memories = stats["sqlite"].get("total_memories", 0)

            self._initialized = True
            logger.info("LTM initialized: %d memories in database", self._total_memories)

        except Exception as e:
            logger.error("LTM initialization failed: %s", e)
            self._proc = None
            # Don't raise — degrade gracefully without LTM

    def system_prompt_block(self) -> str:
        """Return static text for the system prompt."""
        if not self._initialized or not self._auto_recall:
            return ""
        if self._total_memories == 0:
            return (
                "# LTM (Long-Term Memory)\n"
                "Active. Database empty — use ltm_add to store permanent memories.\n"
                "Use ltm_search to search the database when you need to recall "
                "past information."
            )
        return (
            f"# LTM (Long-Term Memory)\n"
            f"Active. {self._total_memories} memories stored with Granite ONNX embeddings.\n"
            f"Use ltm_search to recall past context, decisions, and preferences.\n"
            f"Use ltm_add to store important permanent information."
        )

    # -- Recall: prefetch / queue_prefetch ------------------------------------

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background Granite ONNX search for the next turn."""
        if not self._auto_recall:
            return
        if not query or len(query.strip()) < 3 or not self._is_alive():
            return

        def _do_prefetch():
            try:
                result = self._send({
                    "cmd": "search",
                    "query": query,
                    "top_k": self._search_top_k,
                })
                if result and "entries" in result and result["entries"]:
                    self._prefetch_cache = self._format_prefetch(result)
                    logger.debug("LTM prefetch cached: %d entries", len(result["entries"]))
                else:
                    # No relevant results — clear cache
                    self._prefetch_cache = ""
            except Exception as e:
                logger.debug("LTM prefetch failed: %s", e)
                # Keep old cache — better stale than empty

        threading.Thread(target=_do_prefetch, daemon=True).start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return cached prefetch results (instant, no subprocess call)."""
        if not self._auto_recall:
            return ""
        return self._prefetch_cache

    def _format_prefetch(self, result: dict) -> str:
        """Format search results as context for the system prompt."""
        entries = result.get("entries", [])
        if not entries:
            return ""
        lines = ["## LTM Recall (Granite ONNX, top results)"]
        for e in entries:
            sim = e.get("similarity", 0)
            title = e.get("title", "Unknown")
            content = e.get("full_content") or e.get("relevant_chunk", "")
            if content:
                # Truncate very long content for context efficiency
                if len(content) > 1000:
                    content = content[:1000] + "..."
                lines.append(f"- [{sim:.2f}] **{title}**: {content}")
        return "\n".join(lines)

    # -- Turn persistence -----------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Buffer the turn in RAM for later extraction (no subprocess call)."""
        self._turn_count += 1
        if user_content and len(user_content) > 20:
            self._turn_buffer.append({
                "user": user_content[:2000],
                "assistant": (assistant_content or "")[:2000],
                "timestamp": datetime.datetime.now().isoformat(),
            })

    # -- Tool schemas and dispatch -------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [LTM_SEARCH_SCHEMA, LTM_ADD_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "ltm_search":
            return self._handle_ltm_search(args)
        elif tool_name == "ltm_add":
            return self._handle_ltm_add(args)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _handle_ltm_search(self, args: dict) -> str:
        """Handle ltm_search tool call — synchronous Granite ONNX search."""
        query = args.get("query", "")
        top_k = int(args.get("top_k", 5))
        category = args.get("category")

        if not query or len(query.strip()) < 2:
            return json.dumps({"error": "query is required and must be at least 2 characters"})

        if not self._is_alive():
            return json.dumps({"error": "LTM subprocess is not running"})

        result = self._send({
            "cmd": "search",
            "query": query,
            "top_k": top_k,
            "category": category,
        })

        if "error" in result:
            return json.dumps(result)

        # Format for the agent
        entries = result.get("entries", [])
        if not entries:
            return json.dumps({"message": "No relevant memories found.", "query": query})

        formatted = {
            "query": query,
            "results_count": len(entries),
            "entries": [
                {
                    "id": e.get("id"),
                    "title": e.get("title"),
                    "category": e.get("category"),
                    "similarity": e.get("similarity"),
                    "content": e.get("full_content") or e.get("relevant_chunk", ""),
                    "updated": e.get("updated"),
                }
                for e in entries
            ],
        }
        return json.dumps(formatted, ensure_ascii=False)

    def _handle_ltm_add(self, args: dict) -> str:
        """Handle ltm_add tool call — synchronous add with Granite embedding."""
        title = args.get("title", "")
        content = args.get("content", "")
        category = args.get("category", "geral")
        tags = args.get("tags", "")

        if not title or not content:
            return json.dumps({"error": "title and content are required"})
        if len(content) < 10:
            return json.dumps({"error": "content too short (minimum 10 characters)"})

        if not self._is_alive():
            return json.dumps({"error": "LTM subprocess is not running"})

        result = self._send({
            "cmd": "add",
            "title": title,
            "content": content,
            "category": category,
            "tags": tags,
            "source": "ltm-provider-tool",
        })

        if "error" not in result:
            self._total_memories += 1
            self._obsidian_dirty = True

        return json.dumps(result, ensure_ascii=False)

    # -- Optional hooks -------------------------------------------------------

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Save conversation about to be discarded, return text for compression LLM."""
        if not messages or not self._is_alive():
            return ""

        # Extract user + assistant messages
        excerpts = self._extract_excerpts(messages)
        if not excerpts:
            return ""

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        combined = "\n\n".join(excerpts)[:8000]  # Cap at 8K chars

        # Save to LTM (async — don't block compression)
        self._send_async({
            "cmd": "add",
            "title": f"Sessão — pré-compactação {timestamp}",
            "content": combined,
            "category": "geral",
            "tags": "auto,pre-compress",
            "source": "ltm-provider-pre-compress",
        })
        self._obsidian_dirty = True

        # Return prompt for the compression LLM — it will do the actual extraction
        return (
            "\n[LTM] As mensagens abaixo foram arquivadas no banco de memória de longo prazo. "
            "Examine cuidadosamente e preserve no resumo qualquer informação importante: "
            "decisões, preferências do usuário, regras, configurações, correções, "
            "estado de projetos. Não omita nada que o usuário esperaria que você lembrasse.\n"
        )

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Save remaining conversation to LTM and export to Obsidian if needed."""
        if not self._is_alive():
            return

        # Skip empty sessions
        if not messages and not self._turn_buffer:
            return

        # Collect excerpts from messages + buffered turns
        excerpts = self._extract_excerpts(messages)
        for turn in self._turn_buffer:
            excerpts.append(f"[user]: {turn['user']}")
            excerpts.append(f"[assistant]: {turn['assistant'][:1000]}")

        if excerpts:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            combined = "\n\n".join(excerpts)[:8000]

            try:
                self._send({
                    "cmd": "add",
                    "title": f"Sessão — encerramento {timestamp}",
                    "content": combined,
                    "category": "geral",
                    "tags": "auto,session-end",
                    "source": "ltm-provider-session-end",
                })
                self._obsidian_dirty = True
            except Exception as e:
                logger.warning("LTM session-end save failed: %s", e)

        # Auto-export to Obsidian if dirty
        if self._obsidian_dirty and self._auto_export:
            try:
                result = self._send({"cmd": "export_obsidian", "path": self._obsidian_vault})
                if "error" not in result:
                    logger.info("LTM exported to Obsidian: %s", result.get("exported", "?"))
                    self._obsidian_dirty = False
            except Exception as e:
                logger.warning("LTM Obsidian export failed: %s", e)

        # Clean up buffer
        self._turn_buffer.clear()
        self._turn_count = 0

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes to LTM."""
        if action != "add" or not content or len(content) < 10:
            return
        if not self._is_alive():
            return

        category = "config" if target == "user" else "geral"
        # Generate title from first meaningful line
        first_line = ""
        for line in content.strip().split("\n"):
            clean = line.strip().lstrip("•-*§→").strip()
            if clean and len(clean) > 5:
                first_line = clean[:70]
                break
        if not first_line:
            first_line = content[:60]

        prefix = "User" if target == "user" else "Labo"

        self._send_async({
            "cmd": "add",
            "title": f"{prefix} — {first_line}",
            "content": content[:4000],
            "category": category,
            "tags": f"mirror-{target}",
            "source": f"ltm-provider-mirror-{target}",
        })
        self._obsidian_dirty = True

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """Handle session switches — flush buffers on reset, clear prefetch cache."""
        if reset:
            # Genuinely new conversation — flush accumulated turn buffer
            if self._turn_buffer and self._is_alive():
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                excerpts = []
                for turn in self._turn_buffer:
                    excerpts.append(f"[user]: {turn['user']}")
                    excerpts.append(f"[assistant]: {turn['assistant'][:1000]}")
                combined = "\n\n".join(excerpts)[:8000]
                self._send_async({
                    "cmd": "add",
                    "title": f"Sessão — reset {timestamp}",
                    "content": combined,
                    "category": "geral",
                    "tags": "auto,session-reset",
                    "source": "ltm-provider-session-reset",
                })
            self._turn_buffer.clear()
            self._turn_count = 0

        # Clear prefetch cache on any session switch
        self._prefetch_cache = ""

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs,
    ) -> None:
        """Observe subagent work — save notable delegation results to LTM."""
        if not task or not result or not self._is_alive():
            return
        if len(result) < 50:
            return  # Trivial results not worth storing

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        combined = f"Task: {task[:1000]}\n\nResult: {result[:4000]}"

        self._send_async({
            "cmd": "add",
            "title": f"Delegation — {timestamp}",
            "content": combined,
            "category": "geral",
            "tags": "auto,delegation",
            "source": "ltm-provider-delegation",
        })

    # -- Shutdown -------------------------------------------------------------

    def shutdown(self) -> None:
        """Terminate the LTM subprocess gracefully."""
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                # Wait up to 5s for graceful shutdown
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
                self._proc.wait(timeout=2)
        self._proc = None
        self._initialized = False
        self._prefetch_cache = ""
        self._turn_buffer.clear()

    # -- Internal helpers -----------------------------------------------------

    def _is_alive(self) -> bool:
        """Check if the subprocess is running and healthy."""
        if not self._proc:
            return False
        if self._proc.poll() is not None:
            # Process has exited — don't try to use it
            return False
        return True

    def _ensure_alive(self) -> bool:
        """Ensure subprocess is running; restart if needed (e.g. idle timeout)."""
        if self._is_alive():
            return True
        # Subprocess died — try to restart
        try:
            self._proc = subprocess.Popen(
                [LTM_VENV_PYTHON, LTM_OPS_SCRIPT],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
            ready_line = self._proc.stdout.readline()
            if not ready_line:
                raise RuntimeError("Restart: subprocess closed before ready")
            ready_data = json.loads(ready_line)
            if not ready_data.get("ready"):
                raise RuntimeError(f"Restart: not ready: {ready_data}")
            logger.info("LTM subprocess auto-restarted")
            self._initialized = True
            # Restore stats
            stats = self._send({"cmd": "stats"})
            if stats and "sqlite" in stats:
                self._total_memories = stats["sqlite"].get("total_memories", 0)
            return True
        except Exception as e:
            logger.error("LTM subprocess auto-restart failed: %s", e)
            self._proc = None
            return False

    def _send(self, data: dict, timeout: float = SUBPROCESS_TIMEOUT) -> dict:
        """Send a command to the subprocess and return the response (thread-safe)."""
        if not self._ensure_alive():
            return {"error": "LTM subprocess not running"}

        line = json.dumps(data, ensure_ascii=False) + "\n"

        with self._lock:
            try:
                self._proc.stdin.write(line)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                return {"error": f"Failed to write to LTM subprocess: {e}"}

            # Wait for response with timeout using select
            try:
                ready, _, _ = select.select([self._proc.stdout], [], [], timeout)
                if not ready:
                    return {"error": "LTM subprocess timeout"}
                response_line = self._proc.stdout.readline()
                if not response_line:
                    return {"error": "LTM subprocess closed connection"}
                return json.loads(response_line)
            except json.JSONDecodeError as e:
                return {"error": f"Invalid JSON from LTM: {e}"}
            except Exception as e:
                return {"error": f"LTM communication error: {e}"}

    def _send_async(self, data: dict) -> None:
        """Fire-and-forget send in a daemon thread (non-blocking)."""
        def _do_send():
            result = self._send(data, timeout=60)  # Longer timeout for embedding
            if "error" in result:
                logger.warning("LTM async send failed: %s", result["error"])
        threading.Thread(target=_do_send, daemon=True).start()

    def _extract_excerpts(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract user + assistant messages from message list."""
        excerpts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            if role in ("user", "assistant") and content and len(content) > 20:
                # Truncate individual messages
                if len(content) > 500:
                    content = content[:500] + "..."
                excerpts.append(f"[{role}]: {content}")
        return excerpts

    def check_alive(self) -> bool:
        """Health check — ping the subprocess."""
        result = self._send({"cmd": "ping"}, timeout=5)
        return result.get("pong", False)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the LTM memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = LTMMemoryProvider(config=config)
    ctx.register_memory_provider(provider)