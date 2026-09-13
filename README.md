# ai-memory-cli - An AI Memory System

**Atomic, file-based long-term memory for AI assistants. One fact, one entry. No database, no daemon, no cloud.**

A hierarchical long-term memory system for AI assistants. Structured facts with hybrid TF-IDF + semantic search, automatic deduplication, memory aging, conflict resolution, and a full audit trail. Zero database. One file. Pure Python.

*Friday* is what I (the author) personally call it; the project name is `ai-memory-cli`. Either works.

## Why this exists

Most memory systems do two things differently:

- **They are servers.** A daemon, a port, a database, hooks, an SDK, a cloud. For a single user with a single assistant, that is a lot of machinery to remember what color your editor theme is.
- **They store blobs.** Entire sessions get captured and distilled later, which means facts are fuzzy, redundant, and hard to audit.

Friday Memory is the opposite: explicit, atomic, and self-contained. Each `remember` call creates exactly one fact. Each fact has a subject, predicate, object, confidence, stability, origin, and a full lineage. You can read the entire memory store with a text editor.

It is the lightweight alternative to memory servers that turn "remember what I said" into a distributed system.

## Why

LLMs forget everything between sessions. Most "memory" solutions are bolted-on chat logs or vector stores that grow without structure. Friday Memory treats assistant memory like a real system: every fact is typed, scored for confidence, decayed over time, deduplicated, and reconciled when it contradicts what came before. The result is a memory that gets *more* reliable the longer it runs.

## Features

- **Hybrid search** - TF-IDF and `all-MiniLM-L6-v2` semantic embeddings, weighted and combined with recency + importance
- **Structured schema** - typed facts (preference, project, identity, goal...) with subject/predicate/object, tags, confidence, stability
- **Deduplication** - exact match, semantic similarity (cosine >= 0.75), and fuzzy Jaccard fallback
- **Memory aging** - confidence decays over time; stale, unreferenced facts archive themselves
- **Conflict resolution** - contradictions are merged, archived, or superseded, never silently coexisting
- **Audit trail** - every create/merge/archive/delete logged with reasons, inspectable via `lineage`
- **Working memory** - session-scoped context with topic decay and promotion to long-term facts
- **Write safety** - file locking, atomic writes, automatic backups, panic protection against empty writes
- **Data integrity** - `integrity check` / `integrity repair` finds orphan embeddings, duplicate ids, and broken audit refs
- **Portable** - `--data-dir` or `FRIDAY_MEMORY_DIR` to point the store anywhere

## Install

```bash
pip install -r requirements.txt
```

The first search will download the `all-MiniLM-L6-v2` model (~80 MB). Everything is optional-graceful: if the model is unavailable, search falls back to pure TF-IDF.

## Quickstart

```bash
# save a fact (structured)
python memory.py remember "Alex prefers dark mode" \
  --type preference --subject Alex --predicate prefers --object "dark mode" \
  --tags "editor,preference" --confidence 0.9 --stability stable

# save a fact (quick - positional text maps to summary)
python memory.py remember "Alex uses VS Code for web dev" --tags editor

# search - hybrid TF-IDF + semantic
python memory.py recall "what editor does Alex use?"

# delete
python memory.py forget fact_1785712143053376700

# audit trail
python memory.py lineage fact_1785712143053376700

# list everything (applies aging decay + stability promotion)
python memory.py list
```

### What you'll see

```text
$ python memory.py remember "Alex prefers dark mode" --type preference --subject Alex --predicate prefers --object "dark mode" --tags "editor,preference" --confidence 0.9 --stability stable
saved fact [fact_1785712143053376700]

$ python memory.py remember "Alex uses VS Code for web dev" --type preference --subject Alex --predicate uses --object "VS Code" --tags editor
saved fact [fact_1785712163249380000]

$ python memory.py recall "what editor does Alex use?"
1. [preference] Alex prefers dark mode [editor, preference]
   [Alex prefers dark mode]
   (preference, stable, conf:0.9, imp:0.01, retr:1x)
   id: fact_1785712143053376700  (2026-08-02)

2. [preference] Alex uses VS Code for web dev [editor]
   [Alex uses VS Code]
   (preference, temporary, conf:0.5, imp:0.01, retr:1x)
   id: fact_1785712163249380000  (2026-08-02)
```

Note the confidence gate: the `stable` fact (0.9) outranks the `temporary` one (0.5) even though both match. Confidence is part of the relevance score, not just a filter.

## Where data lives

| Setting | Location |
|---------|----------|
| Default | `~/.config/friday/memory/` |
| `FRIDAY_MEMORY_DIR` env var | `<env>/data/` |
| `--data-dir <path>` flag | `<path>/data/` |

All files are plain JSON:

| File | Contents |
|------|----------|
| `data/facts.json` | the memory store (facts + conversation summaries) |
| `data/embeddings.json` | cached embedding vectors |
| `data/audit.json` | full operation log with reasons |
| `data/working_memory.json` | session-scoped active context |
| `data/backups/` | rolling backups (last 20 kept) + panic snapshots |

## Memory schema

```json
{
  "id": "fact_1785712143053376700",
  "type": "preference",
  "category": "",
  "subject": "Alex",
  "predicate": "prefers",
  "object": "dark mode",
  "summary": "Alex prefers dark mode",
  "details": {},
  "source": {
    "origin": "conversation",
    "timestamp": "2026-08-02T17:15:43+00:00"
  },
  "memory_properties": {
    "confidence": 0.9,
    "importance": 0.0,
    "stability": "stable",
    "plural": false,
    "salience": {
      "base_importance": 0.0,
      "retrieval_count": 1,
      "last_retrieved": "2026-08-02T17:15:43+00:00",
      "last_confirmed": "2026-08-02T17:15:43+00:00",
      "conversation_references": 0,
      "decay_rate": 1.0
    }
  },
  "retrieval": {
    "tags": ["editor", "preference"]
  },
  "last_updated": "2026-08-02T17:15:43+00:00",
  "update_count": 1
}
```

### Stability ladder

Facts climb from `temporary` to `permanent` as they get confirmed:

| Update count | Stability |
|--------------|-----------|
| 1 | temporary |
| 3 | evolving |
| 5 | stable |
| 10 | permanent |

`permanent` facts are exempt from aging. `archived` facts are excluded from retrieval unless `--include-archived`. `quarantine` is used for auto-extracted drafts awaiting confirmation.

## How it works

### Search scoring

`score = TF-IDF(0.25) + semantic(0.55) + recency(0.15) + importance(0.05)`, then multiplied by a confidence factor `(0.5 + conf * 0.5)`. Recency uses a 90-day exponential decay.

### Deduplication

On `remember`, duplicates are resolved in order:
1. **Exact** - same `(subject, predicate, object)`
2. **Semantic** - embedding cosine >= 0.75
3. **Fuzzy** - Jaccard token similarity >= 0.80 (only when no embeddings available)

Duplicates **merge**: confidence bumps, tags union, stability promotes. The audit log records every merge.

### Conflict resolution

Contradictory facts on the same `(subject, predicate)` can't coexist. The system picks a resolution by strength (confidence + origin + stability):

- **merged** - one subsumes the other
- **archived** - weaker inferred fact is archived
- **rejected** - incoming inferred fact is refused
- **superseded** - preference evolution; the old fact is marked `historical` with a `superseded_by` pointer

### Aging

Temporary/evolving facts decay 0.2% confidence per day (scaled by retrieval frequency). Unreferenced temporary facts archive at 90 days; any non-permanent fact with `update_count <= 1` archives at 365 days. Identity facts decay only when unconfirmed for 60+ days.

## What makes it different

### Atomic facts, not blobs
One concept = one entry. A fact is a first-class object with typed fields, not a paragraph your assistant may or may not parse correctly. Plural facts (`user has 3 dogs`) coexist correctly with singular ones; contradictory facts are resolved, never silently stacked.

### A real lifecycle
- **Confidence** decays 0.2%/day for temporary and evolving facts. Stale facts are deprioritized at 180 days, archived at 365.
- **Promotion**: a fact confirmed enough times upgrades temporary → evolving → stable → permanent. Nothing stays a guess forever, and nothing gets to claim permanence without evidence.
- **Quarantine** exists for low-quality or contradictory inputs before they pollute the store.
- **Conflict resolution**: contradictions on the same `(subject, predicate)` are merged, archived, or downgraded. No silent coexistence.

### Everything is audited
Every create, update, merge, archive, and delete is logged with a reason and source IDs. `lineage` shows the full history of a single fact. You can prove where any belief came from.

### Dedup that actually works
`remember` checks, in order: exact `(subject, predicate, object)` match, semantic cosine >= 0.75, and Jaccard >= 0.80. A match means **merge**, never a second copy. Confidence bumps, tags union, stability promotes.

### Working memory, not just long-term
`focus` maintains a separate active-context layer: topics you're actively working on, which decay when ignored and promote into long-term facts when they stick. Useful for assistants that need to know what you're doing *right now* without polluting the permanent store.

### Own your data
Everything lives in `~/.config/friday/memory/data/`:
- `facts.json` — the memory store
- `conversations.json` — session summaries (separate from facts, by design)
- `audit.json` — every mutation
- `embeddings.json`, `tfidf_cache.json` — retrieval caches

Copy the folder, and the assistant's memory moves with it. No export API needed.

## CLI reference

| Command | Description |
|---------|-------------|
| `remember <text>` | Save a fact (structured or quick) |
| `recall <query>` | Hybrid semantic search |
| `forget <id>` | Delete a memory |
| `list` | List stored facts / conversations |
| `save-conv` | Save a conversation summary |
| `consolidate --cluster` | Synthesize concept facts from clusters |
| `consolidate --extract` | Scan conversations for extraction candidates |
| `lineage <id>` | Show full audit trail |
| `warm` | Preload the embedding model |
| `focus <topic>` | Manage working-memory context |
| `integrity check` | Find orphan embeddings, dup ids, broken refs |
| `integrity repair` | Auto-fix everything `check` finds |
| `backup` | Create a timestamped full backup |
| `restore` | Restore from a backup |

## How retrieval works

Queries are scored by a weighted hybrid:

1. **TF-IDF cosine similarity** over summary and detail fields (cached, zero-cost)
2. **Semantic embedding cosine similarity** via sentence-transformers (`all-MiniLM-L6-v2`, ~80MB, local)
3. Combined score, then filtered by confidence threshold (0.5 general, 0.7 strict), staleness, and archive status

Searching is semantic: `recall "database performance problem"` can surface a memory saved as "N+1 query fix". No keyword matching required.

## Requirements

- Python 3.10+
- `sentence-transformers` (see `requirements.txt`; model downloads on first `warm`/`recall`)

## Design notes / tradeoffs

- **Single-user by design.** The write lock is file-based and short-lived. If you need multi-process concurrent writes at scale, this is not the tool.
- **Inferred facts are capped.** Anything derived rather than stated starts at confidence <= 0.69 and can never promote without explicit user confirmation. The system does not let guesses masquerade as facts.
- **Privacy-first.** No telemetry, no cloud, no network calls beyond the model download.

## Architecture & Decisions

**Why JSON files, not SQLite:** Portability and auditability beat query speed at this scale (<10k facts). Every file is human-readable, `git diff`-able, and moves by copying a folder. If the store ever exceeds 50k facts, SQLite would win on indexing — documented here as the migration path.

**Why hybrid TF-IDF + embeddings, not pure vectors:** TF-IDF is free, deterministic, and handles exact terms (IDs, codes) where embeddings blur. Embeddings handle paraphrase. Weighted 0.25/0.55 gives both a voice, verified by `tests/test_memory.py:79 fuzzy_duplicate_merges` passing in TF-IDF-only mode (`FRIDAY_MEMORY_NO_EMBED=1`).

**Why atomic facts:** One fact = one lifecycle. Blobs make aging and conflict resolution impossible — you cannot decay half a paragraph. The schema `type/subject/predicate/object` in `memory.py:1183` enforces this.

**What I learned:** File locking on Windows (`_write_lock:102`) is racy without `mkdir` atomicity; panic protection (`_save_json:140`) prevented a zero-byte `facts.json` loss during a power cut. Confidence as part of score (`memory.py:596`) matters more than as filter — a `permanent` 0.9 fact must outrank a `temporary` 0.9-looking match.

**Next:** Add lemmatization to `_tokenize:402` for better recall, then a second project that is explicitly CompLing (tokenizer from scratch) to show progression `math → software → linguistics → NLP`.

## License

[Custom — Attribution Required, No Resale as Your Own](LICENSE) — do what you want with it, but credit Matt Burke (emjay045) and don't sell it as your own. See `LICENSE` for full terms. Closest standard: `CC-BY-NC-4.0` with resale clarity.
