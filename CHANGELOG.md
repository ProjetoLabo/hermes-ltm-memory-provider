# Changelog

## [1.1.2] — 2026-07-09

### Added
- Hybrid search: combines semantic (sqlite-vec) + lexical (FTS5/BM25) with RRF merge.
- New `chunks_fts` virtual table for full-text indexing via FTS5.
- New `hybrid_search()` function that fuses results from both methods via Reciprocal Rank Fusion.
- New `search_fts()` with automatic LIKE fallback when FTS5 is unavailable.
- Auto-migration: `init_db()` detects existing chunks without FTS5 index and rebuilds automatically on first startup.

### Changed
- `search_memory()` — now uses hybrid search instead of semantic-only.
- `query_context()` — main route used by Hermes Agent now also uses hybrid search.
- `add_memory()`, `update_memory()`, `delete_memory()`, `reindex_all()` — now sync the FTS5 index alongside the vector index.

### Migration
- **Zero manual intervention.** First startup after upgrading creates the `chunks_fts` table and populates the index automatically via `init_db()`.
- If the system's SQLite build lacks FTS5 support, the system degrades gracefully to the previous behavior (semantic-only search).

### Notes
- Removed `test_hybrid_search.py` from the production directory (development artifact with 53 validation tests).

## [1.1.1] — 2026-07-03
- Auto-restart documentation for LTM subprocess.
- Ensure all hook methods use `_ensure_alive` instead of `_is_alive`.

## [1.1.0] — 2026-06-25
- Add `auto_recall` flag for token-saving control over context injection.

## [1.0.0] — 2026-06-01
- Initial release: LTM MemoryProvider with Granite-97m ONNX embeddings + SQLite-vec.
