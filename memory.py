#!/usr/bin/env python3
# Friday memory. Made by me, 03/JUN/2026 <3

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

def _default_memory_dir():
    env = os.environ.get("FRIDAY_MEMORY_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".config" / "friday" / "memory"


MEMORY_DIR = _default_memory_dir()
DATA_DIR = MEMORY_DIR / "data"
FACTS_FILE = DATA_DIR / "facts.json"
CONVERSATIONS_FILE = DATA_DIR / "conversations.json"
AUDIT_LOG = DATA_DIR / "audit.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.json"
WORKING_MEMORY_FILE = DATA_DIR / "working_memory.json"
TFIDF_CACHE_FILE = DATA_DIR / "tfidf_cache.json"
HEALTHY_SNAPSHOT = DATA_DIR / "facts.json.healthy"

_EMBEDDING_CACHE = None
_WRITE_LOCK_DIR = None

CONFIDENCE_THRESHOLD_GENERAL = 0.5
CONFIDENCE_THRESHOLD_STRICT = 0.7
STALE_DAYS = 180
ARCHIVE_DAYS = 365
INFERRED_CONFIDENCE_CAP = 0.69
MIN_CONFIDENCE_THRESHOLD = 0.15
LOW_QUALITY_CONFIDENCE = 0.2
LOW_QUALITY_DAYS = 30
IDENTITY_DRIFT_DAYS = 90
IDENTITY_CONFIRM_DAYS = 60
REDUNDANCY_THRESHOLD = 0.95
MAX_AUTO_EXTRACT = 5
DEFAULT_CONFIDENCE = 0.5
SCHEMA_VERSION = 1

PLURAL_PREDICATES = {
    'likes', 'dislikes', 'loves', 'hates', 'prefers', 'enjoys',
    'plays', 'uses', 'owns', 'has', 'wants', 'needs',
    'knows', 'speaks', 'codes_in', 'works_on',
    'visits', 'has_visited', 'has_played', 'has_read', 'has_watched', 'has_worked_on',
    'listens_to',
}

STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare',
    'ought', 'used', 'it', 'its', "it's", 'i', 'you', 'he', 'she', 'we',
    'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
    'our', 'their', 'this', 'that', 'these', 'those', 'what', 'which',
    'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'because',
    'as', 'until', 'while', 'about', 'between', 'through', 'during',
    'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
    "didn't", "don't", "doesn't", "isn't", "aren't", "wasn't",
    "weren't", "haven't", "hasn't", "hadn't", "won't", "wouldn't",
    "couldn't", "shouldn't", 'let', 'get', 'got', 'gotten', 'make',
    'made', 'said', 'say', 'says', 'going', 'go', 'went', 'gone', 'come',
    'came', 'take', 'took', 'taken', 'like', 'want', 'know', 'think',
    'see', 'use', 'used', 'using', 'done', 'doing', 'does', 'got',
    'well', 'back', 'also', 'ever', 'much', 'still', 'even', 'yet',
    'already', 'though', 'although', 'since', 'any', 'anything',
    'something', 'nothing', 'everything', 'thing', 'things', 'way',
    'many', 'lot', 'really', 'quite', 'actually', 'basically', 'pretty',
    'probably', 'maybe', 'perhaps', 'anyway', 'though', 'however',
    'therefore', 'thus', 'hence', 'indeed', 'instead', 'either',
    'neither', 'whether', 'whatever', 'whoever', 'whenever', 'wherever',
    'however', 'forever', 'always', 'never', 'sometimes', 'often',
    'rarely', 'usually', 'typically', 'generally', 'especially',
}

_EMBEDDER = None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _ts():
    return int(time.time())


@contextlib.contextmanager
def _write_lock(timeout=5):
    global _WRITE_LOCK_DIR
    lock_dir = DATA_DIR / ".write_lock"
    start = time.time()
    while True:
        try:
            lock_dir.mkdir(parents=True, exist_ok=False)
            _WRITE_LOCK_DIR = lock_dir
            break
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError("Could not acquire write lock — another process may be writing")
            time.sleep(0.1)
    try:
        yield
    finally:
        try:
            lock_dir.rmdir()
        except Exception:
            pass
        _WRITE_LOCK_DIR = None


def _load_json(path):
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    if isinstance(obj, dict) and 'items' in obj:
        return obj['items']
    return obj


def _save_json(path, data, allow_empty=False):
    path.parent.mkdir(parents=True, exist_ok=True)

    # Guard: never write empty facts.json if disk has data (unless intentional)
    if not allow_empty and path.name == 'facts.json' and not data:
        if path.exists() and path.stat().st_size > 50:
            panic_dir = path.parent / 'backups'
            panic_dir.mkdir(exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            bname = panic_dir / f'panic_{stamp}.json.backup'
            try:
                shutil.copy2(path, bname)
            except OSError:
                pass
            print(f"PANIC: refusing to write empty facts.json — backed up current state to {bname.name}", file=sys.stderr)
            return

    backup_dir = path.parent / 'backups'
    if path.name == 'facts.json':
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bpath = backup_dir / f'facts_{timestamp}.json.backup'
        try:
            if path.exists():
                shutil.copy2(path, bpath)
        except OSError:
            pass
        try:
            backups = sorted([
                p for p in backup_dir.iterdir()
                if p.suffix == '.backup' and not p.name.startswith('panic_')
            ])
            while len(backups) > 20:
                backups[0].unlink()
                backups.pop(0)
        except OSError:
            pass

    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({"schema_version": SCHEMA_VERSION, "items": data}, f, indent=2, ensure_ascii=False)
    tmp.replace(path)

    # Keep a healthy snapshot after every successful write
    if path.name == 'facts.json' and data:
        try:
            shutil.copy2(path, HEALTHY_SNAPSHOT)
        except OSError:
            pass


def _load_embeddings():
    global _EMBEDDING_CACHE
    if _EMBEDDING_CACHE is not None:
        return _EMBEDDING_CACHE
    if not EMBEDDINGS_FILE.exists():
        _EMBEDDING_CACHE = {}
        return _EMBEDDING_CACHE
    with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
        _EMBEDDING_CACHE = json.load(f)
        return _EMBEDDING_CACHE


def _save_embeddings(data):
    global _EMBEDDING_CACHE
    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = EMBEDDINGS_FILE.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'), ensure_ascii=False)
    tmp.replace(EMBEDDINGS_FILE)
    _EMBEDDING_CACHE = data


def _get_embedding(fact_id):
    store = _load_embeddings()
    return store.get(fact_id)


def _set_embedding(fact_id, vector):
    store = _load_embeddings()
    store[fact_id] = vector
    _save_embeddings(store)


def _delete_embedding(fact_id):
    store = _load_embeddings()
    store.pop(fact_id, None)
    _save_embeddings(store)


# --- working memory: what's on the table right now, session-scoped ---

SESSION_IDLE_MINUTES = 30


def _load_working_memory():
    if not WORKING_MEMORY_FILE.exists():
        return _new_session()
    with open(WORKING_MEMORY_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    now = datetime.now(timezone.utc)
    if not data.get('session_id'):
        return _new_session()
    last_time = data.get('last_query_time')
    if last_time:
        try:
            idle = (now - datetime.fromisoformat(last_time)).total_seconds() / 60
            if idle > SESSION_IDLE_MINUTES:
                return _new_session()
        except Exception:
            pass
    data.setdefault('session_start', data.get('session_id', _now()))
    data.setdefault('recent_queries', [])
    data.setdefault('active', [])
    return data


def _new_session():
    now = _now()
    return {
        "session_id": f"sess_{_ts()}",
        "session_start": now,
        "last_query_time": now,
        "active": [],
        "recent_queries": [],
    }


def _save_working_memory(data):
    data['last_query_time'] = _now()
    WORKING_MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = WORKING_MEMORY_FILE.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(WORKING_MEMORY_FILE)


def _bump_topic(data, topic, entities=None):
    now_str = _now()
    for t in data['active']:
        if t['topic'].lower() == topic.lower():
            t['last_bumped'] = now_str
            t['bump_count'] = t['bump_count'] + 1
            t['decay_count'] = 0
            increment = 0.2 * (0.7 ** (t['bump_count'] - 1))
            t['relevance'] = min(0.9, t['relevance'] + increment)
            if entities:
                existing = set(e.lower() for e in t.get('entities', []))
                for e in entities:
                    if e.lower() not in existing:
                        t['entities'].append(e)
                        existing.add(e.lower())
            return
    data['active'].append({
        "topic": topic,
        "entities": [e for e in (entities or [])],
        "last_bumped": now_str,
        "bump_count": 1,
        "decay_count": 0,
        "relevance": 0.8,
    })


def _decay_working_memory(data):
    kept = []
    for t in data['active']:
        t['decay_count'] = t.get('decay_count', 0) + 1
        if t['decay_count'] >= 10:
            t['relevance'] = 0.0
        else:
            t['relevance'] = round(max(0.0, t['relevance'] - 0.2), 2)
        if t['relevance'] >= 0.1:
            kept.append(t)
    data['active'] = kept


def _get_active_context(data):
    parts = []
    for t in data['active']:
        if t['relevance'] < 0.3:
            continue
        parts.append(t['topic'])
        parts.extend(t.get('entities', []))
    return ' '.join(parts) if parts else ''


def _add_recent_query(data, query):
    data.setdefault('recent_queries', [])
    data['recent_queries'].append(query)
    data['recent_queries'] = data['recent_queries'][-5:]


PROMOTE_BUMP_THRESHOLD = 5
PROMOTE_RELEVANCE_THRESHOLD = 0.6


def _promote_working_memory(wm, facts, embeddings):
    promoted = 0
    now_str = _now()
    for t in wm['active']:
        if t['bump_count'] < PROMOTE_BUMP_THRESHOLD:
            continue
        if t['relevance'] < PROMOTE_RELEVANCE_THRESHOLD:
            continue
        topic = t['topic']
        entities = t.get('entities', [])
        obj_parts = [topic]
        obj_parts.extend(e for e in entities if e.lower() not in topic.lower())
        obj_str = ', '.join(obj_parts)

        candidate = {
            "id": f"fact_{time.time_ns()}",
            "type": "concept",
            "category": "auto",
            "subject": "user",
            "predicate": "related_to",
            "object": obj_str,
            "summary": f"Auto-promoted: {topic}",
            "details": {"promoted_from": topic, "entities": entities},
            "source": {"origin": "inferred", "timestamp": now_str},
            "memory_properties": {
                "confidence": 0.3,
                "importance": 0.3,
                "stability": "quarantine",
            },
            "retrieval": {"tags": ["auto-promoted", topic]},
            "last_updated": now_str,
            "update_count": 1,
        }
        _init_salience(candidate['memory_properties'], 0.3)
        emb = _compute_embedding(candidate['summary'])

        dup, match_type = _find_duplicate(candidate, facts, emb)
        if dup:
            t['bump_count'] = 0
            continue

        facts.append(candidate)
        if emb:
            embeddings[candidate['id']] = emb
        _log_operation('created', f'auto-promoted from working memory topic: {topic}', [candidate['id']])
        promoted += 1
        t['bump_count'] = 0

    return promoted


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        if os.environ.get("FRIDAY_MEMORY_NO_EMBED"):
            _EMBEDDER = False
        else:
            try:
                from sentence_transformers import SentenceTransformer
                _EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                _EMBEDDER = False
    return _EMBEDDER if _EMBEDDER is not False else None


def _compute_embedding(text):
    model = _get_embedder()
    if model is None:
        return None
    emb = model.encode(text, normalize_embeddings=True)
    return emb.tolist()


def _cosine_sim_vec(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0


def _tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _compute_tf(tokens):
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    length = len(tokens)
    if length > 0:
        for t in tf:
            tf[t] /= length
    return tf


def _compute_idf(documents):
    n = len(documents)
    idf = {}
    for doc in documents:
        seen = set(doc)
        for term in seen:
            idf[term] = idf.get(term, 0) + 1
    for term, count in idf.items():
        idf[term] = math.log((n + 1) / (count + 1)) + 1
    return idf


def _cosine_similarity(vec1, vec2):
    dot = 0
    for term, val in vec1.items():
        if term in vec2:
            dot += val * vec2[term]
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)


def _get_search_text(item):
    if 'content' in item:
        return item['content']
    if 'summary' in item:
        title = item.get('title', item.get('type', ''))
        decisions = item.get('decisions', [])
        obj_val = item.get('object', '')
        if isinstance(obj_val, list):
            obj_text = ' '.join(obj_val)
        else:
            obj_text = obj_val
        extra = ' '.join([item.get('subject', ''), item.get('predicate', ''), obj_text])
        return f"{title} {item['summary']} {extra} {' '.join(decisions)}"
    return ''


# audit trail. every mutation gets logged so you can trace where a belief came from, or see where the AI fucked up.

def _log_operation(operation, reason, source_ids):
    log = _load_json(AUDIT_LOG)
    entry = {
        "operation": operation,
        "reason": reason,
        "source_ids": source_ids,
        "timestamp": _now(),
    }
    log.append(entry)
    _save_json(AUDIT_LOG, log)


#
# retrieval = tfidf + embeddings, because either one alone misses stuff
#

_TFIDF_CACHE = None


def _tfidf_cache_valid(items):
    if _TFIDF_CACHE is None:
        return False
    h = hashlib.md5('|'.join(
        item['id'] + str(item.get('update_count', item.get('message_count', 0)))
        for item in items
    ).encode()).hexdigest()
    return _TFIDF_CACHE.get('hash') == h


def _build_tfidf_cache(items):
    global _TFIDF_CACHE
    docs = [_tokenize(_get_search_text(item)) for item in items]
    idf = _compute_idf(docs)
    doc_tokens = docs
    h = hashlib.md5('|'.join(
        item['id'] + str(item.get('update_count', item.get('message_count', 0)))
        for item in items
    ).encode()).hexdigest()
    cache = {'hash': h, 'idf': idf, 'doc_tokens': doc_tokens}
    try:
        with _write_lock():
            tmp = TFIDF_CACHE_FILE.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False)
            tmp.replace(TFIDF_CACHE_FILE)
    except Exception:
        pass
    _TFIDF_CACHE = cache


def _load_tfidf_cache(items):
    global _TFIDF_CACHE
    if _tfidf_cache_valid(items):
        return _TFIDF_CACHE
    if TFIDF_CACHE_FILE.exists():
        try:
            with open(TFIDF_CACHE_FILE, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            h = hashlib.md5('|'.join(
                item['id'] + str(item.get('update_count', item.get('message_count', 0)))
                for item in items
            ).encode()).hexdigest()
            if cached.get('hash') == h:
                _TFIDF_CACHE = cached
                return _TFIDF_CACHE
        except Exception:
            pass
    _build_tfidf_cache(items)
    return _TFIDF_CACHE


def _invalidate_tfidf_cache():
    global _TFIDF_CACHE
    _TFIDF_CACHE = None
    try:
        TFIDF_CACHE_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _score_tfidf(items, query):
    if not items or not query:
        return [0.0] * len(items)
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [0.0] * len(items)
    cache = _load_tfidf_cache(items)
    idf = cache['idf']
    doc_tokens = cache['doc_tokens']
    query_tf = _compute_tf(query_tokens)
    query_vec = {t: query_tf.get(t, 0) * idf.get(t, 0) for t in query_tf}
    scores = []
    for tokens in doc_tokens:
        doc_tf = _compute_tf(tokens)
        doc_vec = {t: doc_tf.get(t, 0) * idf.get(t, 0) for t in doc_tf}
        scores.append(_cosine_similarity(query_vec, doc_vec))
    return scores


def _score_embeddings(items, query):
    q_emb = _compute_embedding(query)
    if q_emb is None:
        return [0.0] * len(items)
    embed_store = _load_embeddings()
    scores = []
    for item in items:
        d_emb = embed_store.get(item.get('id'))
        if d_emb:
            scores.append(_cosine_sim_vec(q_emb, d_emb))
        else:
            scores.append(0.0)
    return scores


def search(items, query, limit=5):
    if not items or not query:
        return []
    tfidf = _score_tfidf(items, query)
    emb = _score_embeddings(items, query)
    now = datetime.now(timezone.utc)
    scored = []
    for i, item in enumerate(items):
        ts_str = item.get('source', {}).get('timestamp', item.get('date', ''))
        if ts_str:
            try:
                age_days = (now - datetime.fromisoformat(ts_str)).total_seconds() / 86400
            except Exception:
                age_days = 0
        else:
            age_days = 0
        recency = math.exp(-age_days / 90)
        importance = _compute_effective_importance(item)
        score = tfidf[i] * 0.25 + emb[i] * 0.55 + recency * 0.15 + importance * 0.05
        confidence = item.get('memory_properties', {}).get('confidence', 0)
        score *= (0.5 + confidence * 0.5)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:limit]]


# filter pass after scoring (confidence and freshness)

def _filter_retrieval(items, include_archived=False, include_stale=False, include_historical=False, strict=False):
    now = datetime.now(timezone.utc)
    threshold = CONFIDENCE_THRESHOLD_STRICT if strict else CONFIDENCE_THRESHOLD_GENERAL
    filtered = []
    for item in items:
        props = item.get('memory_properties', {})

        stability = props.get('stability', 'temporary')
        if stability in ('archived', 'quarantine') and not include_archived:
            continue

        if props.get('historical') and not include_historical:
            continue

        # Items without memory_properties (conversations, legacy flat facts)
        # carry no confidence signal, so treat them as passing.
        if not props:
            filtered.append((0.0, item))
            continue

        confidence = props.get('confidence', 0.0)
        if confidence < threshold:
            continue

        ts_str = item.get('source', {}).get('timestamp', '')
        if ts_str:
            try:
                age_days = (now - datetime.fromisoformat(ts_str)).total_seconds() / 86400
            except Exception:
                age_days = 0
        else:
            age_days = 0

        update_count = item.get('update_count', 1)
        if update_count == 1 and age_days > STALE_DAYS and not include_stale:
            continue

        importance = _compute_effective_importance(item)
        filtered.append((importance, item))

    filtered.sort(key=lambda x: -x[0])
    return [item for _, item in filtered]


# dedupe. if it's basically the same fact again, merge instead of stacking copies.

DEDUP_THRESHOLD = 0.75


def _find_duplicate(new_fact, facts, new_embedding=None):
    new_subj = new_fact.get('subject', '')
    new_pred = new_fact.get('predicate', '')
    new_obj = new_fact.get('object', '')
    new_emb = new_embedding or _get_embedding(new_fact.get('id'))

    for existing in facts:
        existing_emb = _get_embedding(existing.get('id'))

        if new_subj and new_pred and new_obj:
            existing_obj = existing.get('object')
            if existing.get('subject') == new_subj and existing.get('predicate') == new_pred:
                if isinstance(existing_obj, list):
                    if new_obj in existing_obj:
                        return existing, 'exact'
                elif existing_obj == new_obj:
                    return existing, 'exact'

        if new_emb and existing_emb:
            sim = _cosine_sim_vec(new_emb, existing_emb)
            if sim >= REDUNDANCY_THRESHOLD:
                return existing, 'redundant'
            if sim >= DEDUP_THRESHOLD:
                return existing, 'semantic'

        if not new_emb and not existing_emb:
            ns = _get_search_text(new_fact)
            es = _get_search_text(existing)
            if ns and es:
                tok_n = set(_tokenize(ns))
                tok_e = set(_tokenize(es))
                if tok_n and tok_e:
                    jaccard = len(tok_n & tok_e) / len(tok_n | tok_e)
                    if jaccard >= 0.80:
                        return existing, 'fuzzy'

    return None, None


def _merge_fact(target, incoming):
    target['update_count'] = target.get('update_count', 0) + 1
    target['last_updated'] = _now()

    existing_tags = set(target.get('retrieval', {}).get('tags', []))
    new_tags = set(incoming.get('retrieval', {}).get('tags', []))
    merged_tags = sorted(existing_tags | new_tags)
    target.setdefault('retrieval', {})['tags'] = merged_tags

    props = target.setdefault('memory_properties', {})
    in_props = incoming.get('memory_properties', {})
    props['confidence'] = round(min(1.0, max(props.get('confidence', 0), in_props.get('confidence', 0)) + 0.05), 2)
    props['importance'] = round(max(props.get('importance', 0), in_props.get('importance', 0)), 2)

    stability_order = {'quarantine': -1, 'temporary': 0, 'evolving': 1, 'stable': 2, 'permanent': 3}
    cur_stab = props.get('stability', 'temporary')
    in_stab = in_props.get('stability', 'temporary')
    if stability_order.get(in_stab, 0) > stability_order.get(cur_stab, 0):
        props['stability'] = in_stab

    if target.get('update_count', 0) >= 3 and props.get('stability') == 'temporary':
        props['stability'] = 'evolving'
    if target.get('update_count', 0) >= 5 and props.get('stability') == 'evolving':
        props['stability'] = 'stable'
    if target.get('update_count', 0) >= 10 and props.get('stability') == 'stable':
        props['stability'] = 'permanent'

    if incoming.get('details') and not target.get('details'):
        target['details'] = incoming['details']

    if target.get('type') == 'identity':
        target.setdefault('memory_properties', {}).setdefault('salience', {})['last_confirmed'] = _now()

    target_sal = target.get('memory_properties', {}).get('salience')
    in_sal = incoming.get('memory_properties', {}).get('salience')
    if target_sal and in_sal:
        target_sal['retrieval_count'] = max(target_sal.get('retrieval_count', 0), in_sal.get('retrieval_count', 0))
        if in_sal.get('last_retrieved'):
            if not target_sal.get('last_retrieved') or in_sal['last_retrieved'] > target_sal['last_retrieved']:
                target_sal['last_retrieved'] = in_sal['last_retrieved']
        target_sal['conversation_references'] = max(target_sal.get('conversation_references', 0), in_sal.get('conversation_references', 0))


# conflicting facts on the same (subject, predicate) can't both sit there quietly.
# one of them has to lose...

_STAB_ORDER = {'quarantine': -1, 'temporary': 0, 'evolving': 1, 'stable': 2, 'permanent': 3}


def _fact_strength(fact):
    props = fact.get('memory_properties', {})
    conf = props.get('confidence', 0.0)
    origin = fact.get('source', {}).get('origin', 'inferred')
    stab = _STAB_ORDER.get(props.get('stability', 'temporary'), 0)
    return conf + (0.2 if origin == 'conversation' else 0) + stab * 0.1


def _is_plural(predicate, incoming_fact):
    if predicate in PLURAL_PREDICATES:
        return True
    if incoming_fact.get('memory_properties', {}).get('plural'):
        return True
    return False


def _resolve_conflict(incoming, facts):
    subj = incoming.get('subject', '')
    pred = incoming.get('predicate', '')
    if not subj or not pred:
        return None, None, None

    obj = incoming.get('object', '')
    for existing in facts:
        if (existing.get('subject') == subj
            and existing.get('predicate') == pred
            and existing.get('memory_properties', {}).get('stability') != 'archived'):

            existing_obj = existing.get('object')
            if isinstance(existing_obj, list):
                if obj in existing_obj:
                    return None, None, None
            elif existing_obj == obj:
                return None, None, None

            if _is_plural(pred, incoming) or existing.get('memory_properties', {}).get('plural'):
                return 'plural_merge', existing['id'], f"plural predicate — appending '{obj}' to existing"

            existing_score = _fact_strength(existing)
            incoming_score = _fact_strength(incoming)
            diff = abs(existing_score - incoming_score)

            existing_origin = existing.get('source', {}).get('origin', 'inferred')
            incoming_origin = incoming.get('source', {}).get('origin', 'inferred')

            if diff >= 0.3:
                if existing_score > incoming_score:
                    _merge_fact(existing, incoming)
                    msg = f"conflict: existing subsumed incoming ({existing_score:.2f} vs {incoming_score:.2f})"
                    return 'merged', existing['id'], msg
                else:
                    incoming['update_count'] = existing.get('update_count', 0) + 1
                    incoming['last_updated'] = _now()
                    _merge_fact(incoming, existing)
                    for i, f in enumerate(facts):
                        if f['id'] == existing['id']:
                            facts[i] = incoming
                            break
                    msg = f"conflict: incoming subsumed existing ({incoming_score:.2f} vs {existing_score:.2f})"
                    return 'absorbed', incoming.get('id', existing['id']), msg
            elif existing_origin == 'inferred' and incoming_origin != 'inferred':
                existing['memory_properties']['stability'] = 'archived'
                msg = f"conflict: archived weaker inferred fact (conf {existing_score:.2f})"
                return 'archived', existing['id'], msg
            elif incoming_origin == 'inferred' and existing_origin != 'inferred':
                msg = f"conflict: rejected lower-confidence inferred fact (conf {incoming_score:.2f})"
                return 'rejected', existing['id'], msg
            else:
                incoming_id = incoming.get('id')
                existing['memory_properties']['historical'] = True
                existing['memory_properties']['superseded_by'] = incoming_id
                incoming['memory_properties']['supersedes'] = existing.get('id')
                if existing.get('type') == 'identity':
                    existing.setdefault('details', {})['drift'] = True
                facts.append(incoming)
                drift_tag = ' (identity drift)' if existing.get('type') == 'identity' else ''
                msg = f"preference evolution: {existing.get('id')} superseded by {incoming_id}{drift_tag}"
                return 'superseded', existing['id'], msg

    return None, None, None


# salience = how much a fact actually matters, separate from raw importance.
# not every stored fact is equally worth surfacing.

def _init_salience(props, importance):
    props.setdefault('salience', {
        "base_importance": importance,
        "retrieval_count": 0,
        "last_retrieved": None,
        "last_confirmed": _now(),
        "conversation_references": 0,
        "decay_rate": 1.0,
    })


def _bump_salience(fact):
    props = fact.get('memory_properties', {})
    sal = props.get('salience')
    if sal:
        sal['retrieval_count'] = sal.get('retrieval_count', 0) + 1
        sal['last_retrieved'] = _now()
        if fact.get('type') == 'identity':
            sal['last_confirmed'] = _now()


def _compute_effective_importance(fact):
    props = fact.get('memory_properties', {})
    sal = props.get('salience')
    if not sal:
        return props.get('importance', 0.0)
    base = sal.get('base_importance', props.get('importance', 0.0))
    rc = sal.get('retrieval_count', 0)
    bonus = min(0.2, rc * 0.01)
    return round(base + bonus, 2)


# ---- aging. memory that's never used slowly like gets forgot, then gets archived. ----

def _apply_aging(facts):
    now = datetime.now(timezone.utc)
    changed = False
    to_delete = set()
    for f in facts:
        props = f.get('memory_properties', {})
        stability = props.get('stability', 'temporary')
        ts_str = f.get('source', {}).get('timestamp', f.get('created'))
        if not ts_str:
            continue
        try:
            age_days = (now - datetime.fromisoformat(ts_str)).total_seconds() / 86400
        except Exception:
            age_days = 0

        if stability in ('temporary', 'evolving'):
            sal = props.get('salience', {})
            rc = sal.get('retrieval_count', 0) if sal else 0
            salience_factor = max(0.2, 1.0 - (rc * 0.01))
            decay = max(0.1, 1.0 - (age_days * 0.002 * salience_factor))
            new_conf = round(props.get('confidence', 0.0) * decay, 2)
            if new_conf != props.get('confidence', 0.0):
                props['confidence'] = new_conf
                changed = True

        if f.get('type') == 'identity' and stability in ('temporary', 'evolving', 'stable'):
            sal = props.get('salience', {})
            confirmed_str = sal.get('last_confirmed', ts_str) if sal else ts_str
            try:
                confirmed_age = (now - datetime.fromisoformat(confirmed_str)).total_seconds() / 86400
            except Exception:
                confirmed_age = 0
            if confirmed_age > IDENTITY_CONFIRM_DAYS:
                identity_decay = max(0.5, 1.0 - (confirmed_age - IDENTITY_CONFIRM_DAYS) * 0.002)
                new_conf = round(props.get('confidence', 0.0) * identity_decay, 2)
                if new_conf != props.get('confidence', 0.0):
                    props['confidence'] = new_conf
                    changed = True

        conf = props.get('confidence', 0.0)
        if stability == 'temporary' and conf < 0.15:
            props['stability'] = 'archived'
            _log_operation('archived', f'auto-archived (confidence {conf:.2f} below threshold)', [f['id']])
            changed = True

        if stability not in ('permanent', 'archived') and f.get('update_count', 1) <= 1:
            if f.get('type') != 'identity':
                if age_days > ARCHIVE_DAYS:
                    props['stability'] = 'archived'
                    _log_operation('archived', f'auto-archived after {int(age_days)} days without update', [f['id']])
                    changed = True
                elif stability == 'temporary' and age_days > 90:
                    props['stability'] = 'archived'
                    _log_operation('archived', f'auto-archived after {int(age_days)} days (unconfirmed temporary)', [f['id']])
                    changed = True

        old_stab = stability
        update_count = f.get('update_count', 0)
        if update_count >= 3 and stability == 'temporary':
            props['stability'] = 'evolving'
        if update_count >= 5 and stability == 'evolving':
            props['stability'] = 'stable'
        if update_count >= 10 and stability == 'stable':
            props['stability'] = 'permanent'
        if props.get('stability') != old_stab:
            changed = True

        if conf < LOW_QUALITY_CONFIDENCE and age_days > LOW_QUALITY_DAYS:
            to_delete.add(f['id'])

    if to_delete:
        facts[:] = [f for f in facts if f['id'] not in to_delete]
        for fid in to_delete:
            _delete_embedding(fid)
            _log_operation('deleted', f'low-quality fact pruned (conf < {LOW_QUALITY_CONFIDENCE} for > {LOW_QUALITY_DAYS} days)', [fid])
        changed = True

    return changed


# integrity, validation shi like that. cheap checks to keep the store from drifting into garbage.

VALID_TYPES = {'preference', 'project', 'relationship', 'workflow', 'event', 'identity', 'goal', 'habit', 'general', 'concept'}
VALID_STABILITIES = {'temporary', 'evolving', 'stable', 'permanent', 'archived', 'quarantine'}
VALID_ORIGINS = {'conversation', 'system', 'user_import', 'inferred'}


def _validate_fact(fact):
    errors = []
    for field in ['id', 'type', 'summary']:
        if not fact.get(field):
            errors.append(f"missing required field: {field}")
    if fact.get('type') and fact['type'] not in VALID_TYPES:
        errors.append(f"invalid type: {fact['type']}")
    props = fact.get('memory_properties', {})
    if props:
        conf = props.get('confidence')
        if conf is not None and not isinstance(conf, (int, float)):
            errors.append("confidence must be a number")
        elif conf is not None and (conf < 0 or conf > 1):
            errors.append(f"confidence out of range [0-1]: {conf}")
        stab = props.get('stability')
        if stab and stab not in VALID_STABILITIES:
            errors.append(f"invalid stability: {stab}")
    source = fact.get('source', {})
    if source:
        origin = source.get('origin')
        if origin and origin not in VALID_ORIGINS:
            errors.append(f"invalid origin: {origin}")
        if not source.get('timestamp'):
            errors.append("missing source.timestamp")
    return errors


def _find_orphan_embeddings(facts, embeddings):
    fact_ids = {f['id'] for f in facts}
    return [eid for eid in embeddings if eid not in fact_ids]


def _find_missing_embeddings(facts, embeddings):
    return [f['id'] for f in facts if f['id'] not in embeddings]


def _find_orphan_audit_entries(audit, fact_ids, conv_ids):
    valid_ids = fact_ids | conv_ids
    orphans = []
    for entry in audit:
        for sid in entry.get('source_ids', []):
            if sid not in valid_ids:
                orphans.append((entry, sid))
    return orphans


def _find_duplicate_ids(facts):
    seen = set()
    dups = []
    for f in facts:
        fid = f.get('id')
        if fid:
            if fid in seen:
                dups.append(fid)
            seen.add(fid)
    return dups


# consolidation: find groups of similar facts, synthesize a higher-level concept out of them. cool idea right?

def _cluster_facts(facts, embeddings, threshold=0.75):
    n = len(facts)
    adj = [[] for _ in range(n)]
    for i in range(n):
        ei = embeddings.get(facts[i]['id'])
        if ei is None:
            continue
        for j in range(i + 1, n):
            ej = embeddings.get(facts[j]['id'])
            if ej is None:
                continue
            sim = _cosine_sim_vec(ei, ej)
            if sim >= threshold:
                adj[i].append(j)
                adj[j].append(i)
    visited = [False] * n
    clusters = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            cluster = []
            while stack:
                v = stack.pop()
                if not visited[v]:
                    visited[v] = True
                    cluster.append(v)
                    for nb in adj[v]:
                        if not visited[nb]:
                            stack.append(nb)
            clusters.append(cluster)
    return clusters


def cmd_consolidate(args):
    facts = _load_json(FACTS_FILE)
    convs = _load_json(CONVERSATIONS_FILE)
    embeddings = _load_embeddings()

    active = [f for f in facts
              if f.get('memory_properties', {}).get('stability') != 'archived'
              and not f.get('memory_properties', {}).get('historical')
              and not f.get('details', {}).get('consolidated_by')]

    if args.cluster and len(active) >= 3:
        clusters = _cluster_facts(active, embeddings)

        concepts_created = 0
        members_tagged = 0

        for cluster_indices in clusters:
            if len(cluster_indices) < 3:
                continue
            cluster_facts = [active[i] for i in cluster_indices]
            by_subject = {}
            for cf in cluster_facts:
                subj = cf.get('subject', '').lower()
                if subj:
                    by_subject.setdefault(subj, []).append(cf)
            for subj, group in by_subject.items():
                if len(group) < 3:
                    continue
                topics = [g.get('object', '') for g in group if g.get('object')]
                if not topics:
                    continue
                types = [g['type'] for g in group]
                common_type = max(set(types), key=types.count) if types else 'concept'
                tag_sets = [set(g.get('retrieval', {}).get('tags', [])) for g in group]
                common_tags = sorted(list(set.intersection(*tag_sets))) if tag_sets else []
                obj_str = f"{subj}: {', '.join(topics[:5])}"
                if len(topics) > 5:
                    obj_str += f" and {len(topics) - 5} more"

                concept = {
                    "id": f"fact_{time.time_ns()}",
                    "type": 'concept',
                    "category": common_type,
                    "subject": subj,
                    "predicate": 'related_to',
                    "object": obj_str,
                    "summary": f"{subj} is associated with: {', '.join(topics[:5])}",
                    "details": {"consolidates": [g['id'] for g in group]},
                    "source": {
                        "origin": "inferred",
                        "timestamp": _now(),
                    },
                    "memory_properties": {
                        "confidence": 0.65,
                        "importance": 0.5,
                        "stability": "evolving",
                        "plural": False,
                    },
                    "retrieval": {
                        "tags": common_tags + ['consolidated'],
                    },
                    "last_updated": _now(),
                    "update_count": 1,
                }
                _init_salience(concept['memory_properties'], 0.5)
                facts.append(concept)
                emb = _compute_embedding(concept['summary'])
                if emb:
                    embeddings[concept['id']] = emb
                for g in group:
                    deps = g.setdefault('details', {})
                    refs = deps.setdefault('consolidated_by', [])
                    if concept['id'] not in refs:
                        refs.append(concept['id'])
                        members_tagged += 1
                concepts_created += 1
                _log_operation('consolidated', f'created concept from {len(group)} facts', [concept['id']] + [g['id'] for g in group])

        _save_json(FACTS_FILE, facts)
        _save_embeddings(embeddings)

        print(f"consolidation: {concepts_created} concept(s) from {members_tagged} member fact(s)")

    if args.extract and convs:
        extracted = 0
        for conv in convs:
            summary = conv.get('summary', '')
            if not summary or len(summary) < 20:
                continue
            tokens = _tokenize(summary)
            freq = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            significant = sorted([(c, t) for t, c in freq.items() if c >= 3 and len(t) > 3], reverse=True)
            if significant:
                candidates = [t for _, t in significant[:5]]
                print(f"  [{conv.get('id','?')}] {conv.get('title','?')}: entities={candidates}")
                extracted += 1
        if extracted == 0:
            print("  no extraction candidates found in conversations")
        else:
            print(f"extraction: {extracted} conversation(s) with candidate entities")

    if not args.cluster and not args.extract:
        print("use --cluster to synthesize concept facts or --extract to scan conversations")

    return 0


# the CLI commands, one function each. when making this, i was using opencode's desktop app, so if you're the same, just make sure you change opencode.json to have these commands

def cmd_remember(args):
    facts = _load_json(FACTS_FILE)

    details = {}
    if args.details:
        try:
            details = json.loads(args.details)
        except json.JSONDecodeError:
            details = {"note": args.details}

    summary = args.summary or args.content or ''

    origin = args.origin or "conversation"
    confidence = args.confidence if args.confidence != 0.0 else DEFAULT_CONFIDENCE

    if origin == 'inferred' and confidence > INFERRED_CONFIDENCE_CAP:
        confidence = INFERRED_CONFIDENCE_CAP

    if confidence < MIN_CONFIDENCE_THRESHOLD and not args.force:
        print(f"rejected — confidence {confidence:.2f} below minimum {MIN_CONFIDENCE_THRESHOLD:.2f} (use --force to override)")
        return 0

    fact = {
        "id": f"fact_{time.time_ns()}",
        "type": args.type or "general",
        "category": args.category or "",
        "subject": args.subject or "",
        "predicate": args.predicate or "",
        "object": args.object or "",
        "summary": summary,
        "details": details,
        "source": {
            "origin": origin,
            "timestamp": _now(),
        },
        "memory_properties": {
            "confidence": confidence,
            "importance": args.importance or 0.0,
            "stability": args.stability or "temporary",
            "plural": bool(args.plural),
        },
        "retrieval": {
            "tags": args.tags.split(',') if args.tags else [],
        },
        "last_updated": _now(),
        "update_count": 1,
    }

    _init_salience(fact['memory_properties'], args.importance or 0.0)

    if fact.get('type') == 'identity':
        fact['memory_properties']['salience']['last_confirmed'] = _now()

    emb = _compute_embedding(summary or _get_search_text(fact)) if summary else None

    dup, match_type = _find_duplicate(fact, facts, emb)
    if match_type == 'redundant':
        print(f"rejected — redundant with [{dup['id']}] (cosine sim >= {REDUNDANCY_THRESHOLD})")
        return 0
    if dup:
        _merge_fact(dup, fact)
        if emb:
            _set_embedding(dup['id'], emb)
        _save_json(FACTS_FILE, facts)
        _log_operation('merged', f'duplicate merged ({match_type})', [dup['id'], fact['id']])
        print(f"merged into [{dup['id']}] ({match_type}, count={dup['update_count']})")
        return 0

    action, target_id, msg = _resolve_conflict(fact, facts)
    if action == 'merged':
        _save_json(FACTS_FILE, facts)
        _log_operation('merged', msg, [target_id, fact['id']])
        print(f"merged into [{target_id}] (conflict resolved)")
        return 0
    elif action == 'absorbed':
        if emb:
            _set_embedding(target_id, emb)
        _save_json(FACTS_FILE, facts)
        _log_operation('merged', msg, [target_id])
        print(f"absorbed existing [{target_id}] (conflict resolved)")
        return 0
    elif action == 'archived':
        facts.append(fact)
        _save_json(FACTS_FILE, facts)
        if emb:
            _set_embedding(fact['id'], emb)
        _log_operation('archived', msg, [target_id])
        _log_operation('created', 'added as replacement for archived fact', [fact['id']])
        print(f"archived [{target_id}] and added new fact [{fact['id']}]")
        return 0
    elif action == 'rejected':
        _save_json(FACTS_FILE, facts)
        _log_operation('rejected', msg, [target_id])
        print(f"rejected incoming fact — existing [{target_id}] takes priority")
        return 0
    elif action == 'superseded':
        _save_json(FACTS_FILE, facts)
        if emb:
            _set_embedding(fact['id'], emb)
        _log_operation('superseded', msg, [target_id, fact['id']])
        print(f"superseded [{target_id}] with [{fact['id']}] (preference evolution)")
        return 0
    elif action == 'plural_merge':
        existing = next((f for f in facts if f['id'] == target_id), None)
        if existing:
            existing_obj = existing.get('object', '')
            new_obj = fact.get('object', '')
            if isinstance(existing_obj, str):
                existing_list = [existing_obj]
            else:
                existing_list = list(existing_obj)
            if new_obj not in existing_list:
                existing_list.append(new_obj)
            existing['object'] = existing_list
            existing.setdefault('memory_properties', {})['plural'] = True
            new_summary = f"{existing.get('summary', '')} {new_obj}"
            emb2 = _compute_embedding(new_summary)
            if emb2:
                _set_embedding(existing['id'], emb2)
            _merge_fact(existing, fact)
            _save_json(FACTS_FILE, facts)
            _log_operation('merged', msg, [target_id, fact['id']])
            print(f"appended '{new_obj}' to [{target_id}] (plural merge, count={existing['update_count']})")
        return 0

    facts.append(fact)
    _save_json(FACTS_FILE, facts)
    if emb:
        _set_embedding(fact['id'], emb)
    _log_operation('created', 'new fact saved', [fact['id']])
    print(f"saved fact [{fact['id']}]")
    return 0


def cmd_recall(args):
    wm = _load_working_memory()
    _add_recent_query(wm, args.query)
    _decay_working_memory(wm)

    context = _get_active_context(wm)
    query = args.query
    if context:
        query = f"{context} {query}"

    facts = _load_json(FACTS_FILE)
    aged = _apply_aging(facts)
    convs = _load_json(CONVERSATIONS_FILE)

    items = []
    if args.type in (None, 'facts'):
        items.extend(('fact', f) for f in facts)
    if args.type in (None, 'conversations'):
        items.extend(('conv', c) for c in convs)

    targets = [item for _, item in items]
    results = search(targets, query, limit=args.limit)

    if args.tag:
        tag_lower = args.tag.lower()
        results = [r for r in results if any(
            t.lower() == tag_lower
            for t in r.get('retrieval', {}).get('tags', []) + r.get('tags', [])
        )]

    results = _filter_retrieval(
        results,
        include_archived=args.include_archived,
        include_stale=args.include_stale,
        include_historical=args.include_historical,
        strict=args.strict,
    )

    non_contradictory_predicates = {'related_to', 'associated_with', 'is_a', 'has'}
    seen_pairs = {}
    for r in results:
        subj = r.get('subject', '')
        pred = r.get('predicate', '')
        obj = r.get('object', '')
        if subj and pred and not isinstance(obj, list) and pred.lower() not in non_contradictory_predicates:
            key = (subj.lower(), pred.lower())
            if key in seen_pairs:
                existing_obj, existing_id = seen_pairs[key]
                if not isinstance(existing_obj, list) and existing_obj != obj:
                    r['_contradicts'] = existing_id
            else:
                seen_pairs[key] = (obj, r['id'])

        if 'memory_properties' in r and 'salience' not in r.get('memory_properties', {}):
            _init_salience(r['memory_properties'], r['memory_properties'].get('importance', 0.0))
        if r.get('type') == 'identity' and r.get('memory_properties', {}).get('salience', {}).get('last_confirmed') is None:
            r['memory_properties']['salience']['last_confirmed'] = _now()
        if 'memory_properties' in r:
            _bump_salience(r)

        summary = r.get('summary', '')[:100].lower()
        for t in wm['active']:
            if t['topic'].lower() in summary or any(e.lower() in summary for e in t.get('entities', [])):
                _bump_topic(wm, t['topic'])
                break

    if not results:
        _save_working_memory(wm)
        print("no relevant memories found")
        return 0

    now_dt = datetime.now(timezone.utc)
    for r in results:
        if r.get('type') == 'identity':
            sal = r.get('memory_properties', {}).get('salience')
            confirmed_str = sal.get('last_confirmed') if sal else None
            if confirmed_str:
                try:
                    confirmed_age = (now_dt - datetime.fromisoformat(confirmed_str)).total_seconds() / 86400
                    if confirmed_age > IDENTITY_DRIFT_DAYS:
                        drift_date = confirmed_str[:10]
                        r['_drift'] = drift_date
                except Exception:
                    pass

    for i, item in enumerate(results, 1):
        if 'content' in item:
            tags = f" [{', '.join(item['tags'])}]" if item.get('tags') else ""
            print(f"{i}. {item['content']}{tags}")
            print(f"   id: {item['id']}  ({item.get('created', '?')[:10]})\n")
        elif 'type' in item and 'summary' in item:
            tags = f" [{', '.join(item['retrieval']['tags'])}]" if item.get('retrieval', {}).get('tags') else ""
            props = item.get('memory_properties', {})
            sal = props.get('salience')
            eff_imp = _compute_effective_importance(item)
            hist = ' [HISTORICAL]' if props.get('historical') else ''
            contra = item.get('_contradicts', '')
            contra_display = f' [CONTRADICTS: {contra}]' if contra else ''
            drift = item.get('_drift', '')
            drift_display = f' [DRIFT: since {drift}]' if drift else ''
            if sal and sal.get('retrieval_count', 0) > 0:
                meta = f"({item['type']}, {props.get('stability','?')}, conf:{props.get('confidence',0):.1f}, imp:{eff_imp:.2f}, retr:{sal.get('retrieval_count',0)}x){hist}{contra_display}{drift_display}"
            else:
                meta = f"({item['type']}, {props.get('stability','?')}, conf:{props.get('confidence',0):.1f}){hist}{contra_display}{drift_display}"
            obj_val = item.get('object', '')
            if isinstance(obj_val, list):
                obj_display = ', '.join(obj_val)
            else:
                obj_display = obj_val
            print(f"{i}. [{item['type']}] {item['summary'][:200]}{tags}")
            if item.get('subject') and item.get('predicate') and obj_display:
                subj = item.get('subject', '')
                pred = item.get('predicate', '')
                if isinstance(item.get('object'), list):
                    print(f"   [{subj} {pred} [{obj_display}]]")
                else:
                    print(f"   [{subj} {pred} {obj_display}]")
            print(f"   {meta}")
            print(f"   id: {item['id']}  ({item['source']['timestamp'][:10]})\n")
        elif 'summary' in item:
            tags = f" [{', '.join(item['tags'])}]" if item.get('tags') else ""
            print(f"{i}. [{item['title']}]{tags}")
            print(f"   {item['summary'][:200]}")
            print(f"   id: {item['id']}  ({item.get('date', '?')[:10]})\n")

    for f in facts:
        f.pop('_contradicts', None)
        f.pop('_drift', None)

    embeddings = _load_embeddings()
    promoted = _promote_working_memory(wm, facts, embeddings)
    if aged or promoted:
        _save_json(FACTS_FILE, facts)
    if promoted:
        _save_embeddings(embeddings)
        print(f"auto-promoted {promoted} topic(s) to draft facts")
    _save_working_memory(wm)
    return 0


def cmd_forget(args):
    removed = False
    for fp in [FACTS_FILE, CONVERSATIONS_FILE]:
        data = _load_json(fp)
        before = len(data)
        matches = [d for d in data if d['id'] == args.id]
        data = [d for d in data if d['id'] != args.id]
        if len(data) != before:
            _save_json(fp, data, allow_empty=True)
            removed = True
            for m in matches:
                _log_operation('deleted', f'forget command', [m['id']])
            print(f"removed {args.id} from {fp.name}")
    if removed:
        _delete_embedding(args.id)
    if not removed:
        print(f"no memory found with id: {args.id}")
        return 1
    return 0


def cmd_list(args):
    if args.type in (None, 'facts', 'all'):
        facts = _load_json(FACTS_FILE)
        aged = _apply_aging(facts)
        migrated = False
        for f in facts:
            if 'memory_properties' in f and 'salience' not in f.get('memory_properties', {}):
                _init_salience(f['memory_properties'], f['memory_properties'].get('importance', 0.0))
                migrated = True
            if f.get('type') == 'identity' and f.get('memory_properties', {}).get('salience', {}).get('last_confirmed') is None:
                f['memory_properties']['salience']['last_confirmed'] = _now()
                migrated = True
        display = facts
        if args.tag:
            display = [f for f in facts if args.tag in f.get('retrieval', {}).get('tags', []) or args.tag in f.get('tags', [])]
        if not args.include_historical:
            display = [f for f in display if not f.get('memory_properties', {}).get('historical')]
        if display:
            print(f"facts ({len(display)}):")
            for f in display:
                if 'type' in f and 'summary' in f:
                    tags = f" [{', '.join(f['retrieval']['tags'])}]" if f.get('retrieval', {}).get('tags') else ""
                    obj_val = f.get('object', '')
                    if isinstance(obj_val, list):
                        obj_display = '[' + ', '.join(obj_val) + ']'
                    else:
                        obj_display = obj_val
                    subj = f.get('subject', '')
                    pred = f.get('predicate', '')
                    if subj and pred and obj_display:
                        extra = f"  [{subj} {pred} {obj_display}]"
                    else:
                        extra = ''
                    sal = f.get('memory_properties', {}).get('salience')
                    sal_display = ''
                    if sal and sal.get('retrieval_count', 0) > 0:
                        sal_display = f" retr:{sal.get('retrieval_count', 0)}x"
                    hist = ' [HISTORICAL]' if f.get('memory_properties', {}).get('historical') else ''
                    print(f"  [{f['id']}] ({f['type']}) {f['summary'][:120]}{extra}{tags}{sal_display}{hist}")
                elif 'content' in f:
                    tags = f" [{', '.join(f['tags'])}]" if f.get('tags') else ""
                    print(f"  [{f['id']}] {f['content'][:120]}{tags}")
        else:
            print("no facts stored")
        if aged or migrated:
            _save_json(FACTS_FILE, facts)

    if args.type in (None, 'conversations', 'all'):
        convs = _load_json(CONVERSATIONS_FILE)
        if args.tag:
            convs = [c for c in convs if args.tag in c.get('tags', [])]
        if convs:
            print(f"\nconversations ({len(convs)}):")
            for c in convs:
                tags = f" [{', '.join(c['tags'])}]" if c.get('tags') else ""
                print(f"  [{c['id']}] {c['title']}  ({c['date'][:10]}){tags}")
        else:
            print("no conversations saved")
    return 0


def cmd_save_conv(args):
    convs = _load_json(CONVERSATIONS_FILE)
    conv = {
        "id": f"conv_{time.time_ns()}",
        "date": _now(),
        "title": args.title,
        "summary": args.summary,
        "decisions": args.decisions.split('||') if args.decisions else [],
        "tags": args.tags.split(',') if args.tags else [],
        "message_count": args.messages or 0,
    }
    convs.append(conv)
    _save_json(CONVERSATIONS_FILE, convs)
    _log_operation('created', 'conversation summary saved', [conv['id']])
    print(f"saved conversation [{conv['id']}]")

    summary_tokens = _tokenize(args.summary or '')
    freq = {}
    for t in summary_tokens:
        freq[t] = freq.get(t, 0) + 1
    significant = sorted([(c, t) for t, c in freq.items() if c >= 2 and len(t) > 3], reverse=True)
    if not significant:
        return 0

    facts = _load_json(FACTS_FILE) if args.auto_extract else None
    embeddings = _load_embeddings() if args.auto_extract else None
    existing_text = ' '.join(_get_search_text(f) for f in (facts or []))
    suggested = 0
    extracted = 0

    for _, term in significant:
        if existing_text and term in existing_text.lower():
            continue
        in_prior_conv = any(term in (c.get('summary', '') or '').lower() for c in convs[:-1])
        if in_prior_conv:
            continue

        if args.auto_extract and facts is not None:
            candidate = {
                "id": f"fact_{time.time_ns()}",
                "type": "concept",
                "category": "auto",
                "subject": "user",
                "predicate": "related_to",
                "object": term,
                "summary": f"Extracted: {term}",
                "details": {"extracted_from": conv['id']},
                "source": {"origin": "inferred", "timestamp": _now()},
                "memory_properties": {
                    "confidence": 0.25,
                    "importance": 0.2,
                    "stability": "quarantine",
                },
                "retrieval": {"tags": ["auto-extracted", term]},
                "last_updated": _now(),
                "update_count": 1,
            }
            _init_salience(candidate['memory_properties'], 0.2)
            if extracted >= MAX_AUTO_EXTRACT:
                continue
            emb = _compute_embedding(candidate['summary'])
            dup, _ = _find_duplicate(candidate, facts, emb)
            if dup:
                continue
            facts.append(candidate)
            if emb:
                embeddings[candidate['id']] = emb
            _log_operation('created', f'auto-extracted from conversation: {term}', [candidate['id']])
            extracted += 1
        else:
            suggested += 1

    if args.auto_extract and extracted:
        _save_json(FACTS_FILE, facts)
        if embeddings:
            _save_embeddings(embeddings)
        print(f"auto-extracted {extracted} fact(s) from conversation")
    elif suggested:
        print(f"suggestion: consider saving facts about: {', '.join(t for _, t in significant[:5])}")
    return 0


def cmd_lineage(args):
    log = _load_json(AUDIT_LOG)
    entries = [e for e in log if args.id in e.get('source_ids', [])]
    if not entries:
        print(f"no audit trail found for: {args.id}")
        return 0
    print(f"lineage for {args.id}:")
    for e in entries:
        print(f"  [{e['timestamp'][:19]}] {e['operation']} — {e['reason']}")
    fact = next((f for f in _load_json(FACTS_FILE) if f['id'] == args.id), None)
    if fact:
        props = fact.get('memory_properties', {})
        if props.get('supersedes'):
            print(f"  supersedes: {props['supersedes']}")
        if props.get('superseded_by'):
            print(f"  superseded_by: {props['superseded_by']}")
        if props.get('historical'):
            print(f"  status: historical")
    return 0


def cmd_warm(args):
    print("Loading embedding model...", end=' ', flush=True)
    model = _get_embedder()
    if model:
        print("ready")
    else:
        print("not available (sentence-transformers missing)")
    return 0


def cmd_focus(args):
    data = _load_working_memory()

    if args.action == 'list':
        if data['active']:
            print("working memory:")
            for t in data['active']:
                entities = ', '.join(t.get('entities', []))
                extra = f" [{entities}]" if entities else ""
                print(f"  {t['topic']} (rel:{t['relevance']}, bumps:{t['bump_count']}){extra}")
        else:
            print("no active working memory")
        if data.get('recent_queries'):
            print(f"\nrecent queries ({len(data['recent_queries'])}):")
            for q in data['recent_queries']:
                print(f"  {q}")
        return 0

    if args.action == 'decay':
        before = len(data['active'])
        _decay_working_memory(data)
        _save_working_memory(data)
        after = len(data['active'])
        print(f"decayed working memory: {before} -> {after} active topics")
        return 0

    if args.action == 'clear':
        data['active'] = []
        data['recent_queries'] = []
        _save_working_memory(data)
        print("working memory cleared")
        return 0

    entities = [e.strip() for e in args.entities.split(',')] if args.entities else []
    topic = args.action
    _bump_topic(data, topic, entities)
    rel = next((t['relevance'] for t in data['active'] if t['topic'].lower() == topic.lower()), 0.8)
    _save_working_memory(data)
    print(f"focused on '{topic}' (rel:{rel})")
    if entities:
        print(f"  entities: {', '.join(entities)}")
    return 0


def cmd_integrity(args):
    facts = _load_json(FACTS_FILE)
    convs = _load_json(CONVERSATIONS_FILE)
    audit = _load_json(AUDIT_LOG)
    embeddings = _load_embeddings()

    issues = []
    fact_ids = {f['id'] for f in facts}
    conv_ids = {c['id'] for c in convs}

    for f in facts:
        for e in _validate_fact(f):
            issues.append(f"[{f.get('id','?')}] {e}")

    dups = _find_duplicate_ids(facts)
    for did in dups:
        issues.append(f"[{did}] duplicate id found")

    orphans = _find_orphan_embeddings(facts, embeddings)
    for oid in orphans:
        issues.append(f"[{oid}] orphan embedding (no matching fact)")

    if _get_embedder() is not None:
        missing = _find_missing_embeddings(facts, embeddings)
        for mid in missing:
            issues.append(f"[{mid}] missing embedding")

    audit_orphans = _find_orphan_audit_entries(audit, fact_ids, conv_ids)
    for entry, sid in audit_orphans:
        issues.append(f"audit entry {entry.get('timestamp','?')[:10]} references non-existent id: {sid}")

    if args.action == 'repair':
        if orphans:
            for oid in orphans:
                embeddings.pop(oid, None)
            _save_embeddings(embeddings)

        if _get_embedder() is not None and missing:
            for mid in missing:
                f = next((x for x in facts if x['id'] == mid), None)
                if f:
                    text = _get_search_text(f)
                    if text:
                        emb = _compute_embedding(text)
                        if emb:
                            embeddings[mid] = emb
            _save_embeddings(embeddings)

        bad = set()
        for i, entry in enumerate(audit):
            for sid in entry.get('source_ids', []):
                if sid not in fact_ids and sid not in conv_ids:
                    bad.add(i)
                    break
        if bad:
            for i in sorted(bad, reverse=True):
                audit.pop(i)
            _save_json(AUDIT_LOG, audit)

        removed_dups = 0
        seen_ids = set()
        deduped = []
        for f in facts:
            if f['id'] in seen_ids:
                removed_dups += 1
            else:
                seen_ids.add(f['id'])
                deduped.append(f)
        if removed_dups:
            facts[:] = deduped
            _save_json(FACTS_FILE, facts)
            _log_operation('repaired', f'removed {removed_dups} duplicate fact(s)', [])

        repaired_missing = len(missing) if _get_embedder() is not None else 0
        print(f"repair: removed {len(orphans)} orphan embedding(s), "
              f"generated {repaired_missing} missing embedding(s), "
              f"removed {len(bad)} orphan audit entry(s)")
        if removed_dups:
            print(f"repair: removed {removed_dups} duplicate fact id(s)")

        issues = []
        fact_ids = {f['id'] for f in facts}
        for f in facts:
            for e in _validate_fact(f):
                issues.append(f"[{f.get('id','?')}] {e}")
        dups = _find_duplicate_ids(facts)
        for did in dups:
            issues.append(f"[{did}] duplicate id found")
        orphans = _find_orphan_embeddings(facts, embeddings)
        for oid in orphans:
            issues.append(f"[{oid}] orphan embedding (no matching fact)")
        if _get_embedder() is not None:
            missing = _find_missing_embeddings(facts, embeddings)
            for mid in missing:
                issues.append(f"[{mid}] missing embedding")
        audit_orphans = _find_orphan_audit_entries(audit, fact_ids, conv_ids)
        for entry, sid in audit_orphans:
            issues.append(f"audit entry {entry.get('timestamp','?')[:10]} references non-existent id: {sid}")

    if not issues:
        print("integrity check passed — no issues found")
        return 0

    print(f"integrity check: {len(issues)} issue(s) found:")
    for issue in issues:
        print(f"  {issue}")
    return 0 if args.action == 'repair' else 1


def cmd_backup(args):
    import shutil
    from datetime import datetime

    backup_dir = MEMORY_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"backup_{stamp}"

    shutil.copytree(DATA_DIR, dest)

    total_size = sum(f.stat().st_size for f in dest.rglob('*') if f.is_file())
    print(f"backup created: {dest}")
    print(f"size: {total_size / 1024:.1f} KB")
    return 0


def cmd_restore(args):
    import shutil

    backup_dir = MEMORY_DIR / "backups"

    if args.list:
        if not backup_dir.exists():
            print("no backups found")
            return 0
        backups = sorted(b for b in backup_dir.iterdir() if b.is_dir() and b.name.startswith('backup_'))
        if not backups:
            print("no backups found")
            return 0
        print("available backups:")
        for b in backups:
            size = sum(f.stat().st_size for f in b.rglob('*') if f.is_file())
            print(f"  {b.name}  ({size / 1024:.1f} KB)")
        return 0

    stamp = args.backup
    if not stamp:
        backups = sorted([b for b in backup_dir.iterdir() if b.is_dir() and b.name.startswith('backup_')], reverse=True)
        if not backups:
            print("no backups found")
            return 1
        src = backups[0]
        stamp = src.name
    else:
        src = backup_dir / f"backup_{stamp}"
        if not src.exists():
            print(f"backup not found: backup_{stamp}")
            return 1

    safety = MEMORY_DIR / "pre_restore_safety"
    if safety.exists():
        shutil.rmtree(safety)
    shutil.copytree(DATA_DIR, safety)

    for child in DATA_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in src.iterdir():
        if child.is_dir():
            shutil.copytree(child, DATA_DIR / child.name)
        else:
            shutil.copy2(child, DATA_DIR / child.name)

    print(f"restored from {stamp}")
    print(f"prior state saved to: {safety}")
    return 0


# entry point

def main():
    parser = argparse.ArgumentParser(description='Friday memory system')
    parser.add_argument('--data-dir', help='override the memory data directory (default: ~/.config/friday/memory, or $FRIDAY_MEMORY_DIR)')
    sub = parser.add_subparsers(dest='command')

    p = sub.add_parser('remember', help='save a fact (structured or quick)')
    p.add_argument('content', nargs='?', default='', help='fact text (maps to summary if --summary not given)')
    p.add_argument('--summary', help='summary text (overrides content)')
    p.add_argument('--type', choices=['preference', 'project', 'relationship', 'workflow', 'event', 'identity', 'goal', 'habit', 'general'], default='general')
    p.add_argument('--category', default='')
    p.add_argument('--subject', default='')
    p.add_argument('--predicate', default='')
    p.add_argument('--object', default='')
    p.add_argument('--details', help='optional context (JSON string or plain text)')
    p.add_argument('--origin', choices=['conversation', 'system', 'user_import', 'inferred'], default='conversation')
    p.add_argument('--confidence', type=float, default=0.0)
    p.add_argument('--importance', type=float, default=0.0)
    p.add_argument('--stability', choices=['temporary', 'evolving', 'stable', 'permanent', 'quarantine'], default='temporary')
    p.add_argument('--tags', help='comma-separated tags')
    p.add_argument('--plural', action='store_true', help='force plural coexistence even for non-standard predicates')
    p.add_argument('--force', action='store_true', help='bypass minimum confidence threshold')

    p = sub.add_parser('recall', help='semantic search of memories')
    p.add_argument('query', help='search query')
    p.add_argument('--limit', type=int, default=5, help='max results')
    p.add_argument('--type', choices=['facts', 'conversations'])
    p.add_argument('--include-archived', action='store_true', help='include archived memories')
    p.add_argument('--include-stale', action='store_true', help='include stale memories')
    p.add_argument('--include-historical', action='store_true', help='include historical (superseded) memories')
    p.add_argument('--strict', action='store_true', help='use strict confidence threshold (0.7)')
    p.add_argument('--tag', help='filter by tag (case-insensitive)')

    p = sub.add_parser('forget', help='delete a memory by id')
    p.add_argument('id', help='memory id (fact_xxx or conv_xxx)')

    p = sub.add_parser('list', help='list stored memories')
    p.add_argument('--tag', help='filter by tag')
    p.add_argument('--include-historical', action='store_true', help='include historical (superseded) memories')
    p.add_argument('--type', choices=['facts', 'conversations', 'all'], default='all')

    p = sub.add_parser('save-conv', help='save a conversation summary')
    p.add_argument('--title', required=True)
    p.add_argument('--summary', required=True)
    p.add_argument('--decisions', help='key decisions (|| separated)')
    p.add_argument('--tags', help='comma-separated tags')
    p.add_argument('--messages', type=int, help='message count')
    p.add_argument('--auto-extract', action='store_true', help='auto-create draft facts from significant terms in summary')

    p = sub.add_parser('lineage', help='show audit trail for a memory')
    p.add_argument('id', help='memory id (fact_xxx or conv_xxx)')

    p = sub.add_parser('consolidate', help='run consolidation/compression pipeline')
    p.add_argument('--cluster', action='store_true', help='synthesize concept facts from fact clusters')
    p.add_argument('--extract', action='store_true', help='scan conversations for extraction candidates')

    p = sub.add_parser('warm', help='preload the embedding model')

    p = sub.add_parser('focus', help='manage working memory context')
    p.add_argument('action', help='topic to focus on, or list/decay/clear')
    p.add_argument('--entities', help='comma-separated related entities')

    p = sub.add_parser('integrity', help='check or repair data integrity')
    p.add_argument('action', choices=['check', 'repair'], help='check (read-only) or repair (auto-fix)')

    p = sub.add_parser('backup', help='create a timestamped backup of all data')

    p = sub.add_parser('restore', help='restore data from a backup')
    p.add_argument('backup', nargs='?', help='backup timestamp to restore (omit for latest)')
    p.add_argument('--list', action='store_true', help='list available backups')

    args = parser.parse_args()

    if args.data_dir:
        global MEMORY_DIR, DATA_DIR, FACTS_FILE, CONVERSATIONS_FILE, AUDIT_LOG, EMBEDDINGS_FILE, WORKING_MEMORY_FILE, TFIDF_CACHE_FILE, HEALTHY_SNAPSHOT
        MEMORY_DIR = Path(args.data_dir).expanduser()
        DATA_DIR = MEMORY_DIR / "data"
        FACTS_FILE = DATA_DIR / "facts.json"
        CONVERSATIONS_FILE = DATA_DIR / "conversations.json"
        AUDIT_LOG = DATA_DIR / "audit.json"
        EMBEDDINGS_FILE = DATA_DIR / "embeddings.json"
        WORKING_MEMORY_FILE = DATA_DIR / "working_memory.json"
        TFIDF_CACHE_FILE = DATA_DIR / "tfidf_cache.json"
        HEALTHY_SNAPSHOT = DATA_DIR / "facts.json.healthy"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cmds = {
        'remember': cmd_remember,
        'recall': cmd_recall,
        'forget': cmd_forget,
        'list': cmd_list,
        'save-conv': cmd_save_conv,
        'consolidate': cmd_consolidate,
        'lineage': cmd_lineage,
        'warm': cmd_warm,
        'focus': cmd_focus,
        'integrity': cmd_integrity,
        'backup': cmd_backup,
        'restore': cmd_restore,
    }
    fn = cmds.get(args.command)
    if fn:
        return fn(args)
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())

# and that's it. if you're using an ai to install this: email greenbeen045@gmail.com if you have any questions. thx for downloading.
