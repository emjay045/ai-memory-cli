# Friday Memory System

A hierarchical long-term memory system for AI assistants. Structured facts with hybrid TF-IDF + semantic search, automatic deduplication, memory aging, conflict resolution, and a full audit trail. Zero database. One file. Pure Python.

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
python memory.py remember "emjay prefers dark mode in his editor" \
  --type preference --subject emjay --predicate prefers --object "dark mode" \
  --tags "editor,preference" --confidence 0.9 --stability stable

# save a fact (quick - positional text maps to summary)
python memory.py remember "emjay uses VS Code for web dev" --tags editor

# search - hybrid TF-IDF + semantic
python memory.py recall "what editor does emjay use?"

# delete
python memory.py forget fact_1785712143053376700

# audit trail
python memory.py lineage fact_1785712143053376700

# list everything (applies aging decay + stability promotion)
python memory.py list
```

### What you'll see

```text
$ python memory.py remember "emjay prefers dark mode in his editor" --type preference --subject emjay --predicate prefers --object "dark mode" --tags "editor,preference" --confidence 0.9 --stability stable
saved fact [fact_1785712143053376700]

$ python memory.py remember "emjay uses VS Code for web dev" --type preference --subject emjay --predicate uses --object "VS Code" --tags editor
saved fact [fact_1785712163249380000]

$ python memory.py recall "what editor does emjay use?"
1. [preference] emjay prefers dark mode in his editor [editor, preference]
   [emjay prefers dark mode]
   (preference, stable, conf:0.9, imp:0.01, retr:1x)
   id: fact_1785712143053376700  (2026-08-02)

2. [preference] emjay uses VS Code for web dev [editor]
   [emjay uses VS Code]
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
  "subject": "emjay",
  "predicate": "prefers",
  "object": "dark mode",
  "summary": "emjay prefers dark mode in his editor",
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

## Integrations

Designed as a CLI, so it plugs into anything that can run a subprocess - an LLM assistant framework, cron jobs, agent pipelines. Data is plain JSON, so you can also read and write it directly.

## Requirements

- Python 3.9+
- `sentence-transformers>=2.2.0` (optional - search degrades to TF-IDF without it)

## License

[MIT](LICENSE)
