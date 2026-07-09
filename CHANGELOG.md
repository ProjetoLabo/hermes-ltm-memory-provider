# Changelog

## [1.1.2] — 2026-07-09

### Added
- Hybrid search: combines semantic (sqlite-vec) + lexical (FTS5/BM25) with RRF merge.
- New `chunks_fts` virtual table for full-text indexing via FTS5.
- New `hybrid_search()` function that fuses results from both methods via Reciprocal Rank Fusion.
- New `search_fts()` with automatic LIKE fallback when FTS5 is unavailable.
- Auto-migration: `init_db()` detects existing chunks without FTS5 index and rebuilds automatically on first startup.
- **Multilingual centroid-based classification**: replaces hardcoded keyword matching with embedding centroids computed from seed phrases in EN, PT, ES, and FR. Works across all languages supported by the Granite-97m model.
- **DB migration**: legacy Portuguese category names (`geral`, `infraestrutura`, `projeto`, `pesquisa`, `decisao`, `correcao`) are auto-translated to English on startup.
- Bilingual Obsidian vault import: detects filenames in both PT and EN.

### Changed
- `classify_category()` — now uses embedding centroid similarity instead of substring keyword matching. No language-specific keyword lists to maintain.
- `search_memory()` — now uses hybrid search instead of semantic-only.
- `query_context()` — main route used by Hermes Agent now also uses hybrid search.
- `add_memory()`, `update_memory()`, `delete_memory()`, `reindex_all()` — now sync the FTS5 index alongside the vector index.
- **Full internationalization**: all CLI output, help text, docstrings, comments, and tool schemas translated to English. Plugin tool schemas now show English category names.
- `CATEGORY_KEYWORDS` replaced by `CATEGORY_SEEDS` with multilingual seed phrases.

### Migration
- **Zero manual intervention.** First startup after upgrading creates the `chunks_fts` table and populates the index automatically via `init_db()`.
- Category values in the database are migrated automatically on first `init_db()` call — no reindex needed.
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
