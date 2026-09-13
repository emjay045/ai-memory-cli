# ai-memory-cli

A simple file-based memory I built for my AI assistant. Each fact is one entry, so no database, no server, just JSON files. One Python file does the whole thing.

I call it Friday; that's what my assistant is named. The package is called `ai-memory-cli` so other people can use it too.

## Why I built this

Most memory stuff out there is overkill, or data-sensitive. It's either a whole server with a database and an SDK, or it just dumps your whole chat history into a blob and hopes for the best. Both felt dumb for a single user with one assistant.

So I made this instead. Every `remember` saves exactly one fact — subject, predicate, object, plus confidence and other things. You can open the JSON and read it yourself. I got tired of the assistant forgetting things, and chat logs weren't cutting it, I wanted facts that get checked, merged, and cleaned up over time.

## What it does

- **Hybrid search** — TF-IDF plus `all-MiniLM-L6-v2` embeddings, plus recency and importance weighting. Either one alone misses stuff, together they actually work.
- **Structured facts** — every fact has a type, subject/predicate/object, tags, confidence, stability. Not just random text.
- **Dedup** — checks exact match, then semantic similarity (cosine >= 0.75), then fuzzy fallback. If it's the same thing, it merges instead of duplicating.
- **Aging** — old unused facts slowly lose confidence and eventually archive themselves so the store doesn't fill with junk.
- **Conflict handling** — if two facts contradict on the same (subject, predicate), it picks one and archives/supersedes the other instead of letting both sit there.
- **Audit trail** — every change gets logged with a reason, you can run `lineage` on any fact and see where it came from.
- **Working memory** — short-term context for what you're doing right now, decays if you stop mentioning it.
- **Write safety** — file locking, atomic writes, backups, and a panic guard so it never wipes your store to an empty file.
- **Integrity checks** — `integrity check` / `repair` catches orphan embeddings, duplicate IDs, broken links.
- **Portable** — point it anywhere with `--data-dir` or `FRIDAY_MEMORY_DIR`, everything's just JSON.

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

I score with `TF-IDF*0.25 + semantic*0.55 + recency*0.15 + importance*0.05`, then scale by confidence `(0.5 + conf*0.5)`. Recency is a 90-day decay so old stuff naturally sinks.

### Deduplication

On `remember` it checks:
1. **Exact** — same `(subject, predicate, object)`
2. **Semantic** — embedding cosine >= 0.75
3. **Fuzzy** — Jaccard >= 0.80 if embeddings aren't available

If it matches, it merges instead of making a copy, it bumps confidence, merges tags, maybe promotes stability.

### Conflict resolution

You can't have two different facts for the same `(subject, predicate)`. It compares strength (confidence + origin + stability) and either merges, archives the weaker inferred one, rejects the new one, or marks the old one as `historical` with a `superseded_by` pointer.

### Aging

Temporary/evolving facts lose 0.2% per day (slower if you retrieve them a lot). Unconfirmed temporary stuff archives at 90 days, anything with `update_count <= 1` at 365 days. Identity facts only decay if unconfirmed for 60+ days.

## What makes it different

### One fact = one entry
Not a big chat dump. Each fact is its own thing with fields, so stuff like `user has 3 dogs` doesn't get confused with singular/plural stuff, and contradictions actually get handled instead of just stacking.

### It ages like real memory
- Temporary/evolving facts lose like 0.2% confidence per day. Old stuff gets pushed down at 180 days, archived at 365 if nobody confirmed it.
- If you confirm a fact enough times it levels up: temporary → evolving → stable → permanent. Nothing stays permanent unless it earned it.
- Low-quality or conflicting stuff goes to quarantine first so it doesn't mess up the main store.

### Everything is logged
Every create/update/merge/archive/delete writes to the audit log with why it happened. `lineage <id>` shows you the whole history for one fact.

### Dedup actually works
Checks exact `(subject, predicate, object)`, then semantic cosine >= 0.75, then Jaccard >= 0.80. If it matches, it merges — bumps confidence, unions tags, promotes stability. No duplicates.

### Working memory vs long-term
`focus` keeps a short-term layer for what you're doing right now. It decays if you ignore it, and if you keep mentioning the same topic it can promote into long-term facts.

### Your data stays yours
Everything is in `~/.config/friday/memory/data/`:
- `facts.json` — the actual memories
- `conversations.json` — chat summaries (kept separate on purpose)
- `audit.json` — the log
- `embeddings.json`, `tfidf_cache.json` — just caches

Copy the folder and you move the whole memory. No export needed.

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

## Notes

- Single-user setup. The write lock is just a folder lock — fine for one person, wouldn't scale to a ton of writers at once.
- Inferred facts are capped at 0.69. If the system guesses something, it can't pretend it's a fact until you confirm it.
- No telemetry, no cloud, no phone-home stuff. Model downloads once, then it's local.

## How I built it

**Why JSON instead of SQLite:** At this size (<10k facts) being able to read the files and `git diff` them matters more than raw speed. You can just copy the folder to move everything. If it ever got to 50k+ I'd switch to SQLite for indexing — left that as a future path.

**Why both TF-IDF and embeddings:** TF-IDF is free and exact (good for IDs, codes), embeddings get paraphrases. Using both weighted 0.25/0.55 worked best — tested with `FRIDAY_MEMORY_NO_EMBED=1` mode and the fuzzy dedup tests still pass.

**Why atomic facts:** One fact = one lifecycle. If you store blobs you can't decay half of it or handle conflicts cleanly. The `type/subject/predicate/object` shape in `memory.py` keeps it strict.

**Things I ran into:** Windows file locking is annoying — `mkdir` as a lock was the only thing that was actually atomic. Added panic protection in `_save_json` after I almost wiped `facts.json` to 0 bytes during a power cut. Also learned that confidence needs to be in the score, not just a filter — otherwise a permanent 0.9 and a temporary 0.9 look the same.

**How I worked:** I built the schema, search, dedup, and aging myself — used autocomplete/AI for boilerplate and to rubber-duck ideas, same as I'd use Stack Overflow. I debugged the file lock and panic guard myself.

**Next up:** Maybe add lemmatization to `_tokenize` so `running` finds `run`, and later a second project that's more linguistics-focused from scratch.

## Known issues / TODO

- `recall` ranking feels a bit off when confidence ties — might tweak recency weight again
- `integrity repair` still a little noisy on fresh stores (false positives for missing embeddings)
- Want to add a quick `friday-memory stats` command

## License

[Custom — Attribution Required, No Resale as Your Own](LICENSE) — do what you want with it, but credit Matt Burke (emjay045) and don't sell it as your own. See `LICENSE` for full terms. Closest standard: `CC-BY-NC-4.0` with resale clarity.
