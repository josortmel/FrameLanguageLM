/**
 * Match parsed FilmAffinity events to catalog items.
 * Uses Spanish title mappings (akas_es.json) + catalog titles + fuzzy matching.
 */

function normalize(s) {
  return s.toLowerCase()
    .normalize('NFD').replace(/[̀-ͯ]/g, '')
    .replace(/[''`]/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

function levenshteinRatio(a, b) {
  const m = a.length, n = b.length;
  if (m === 0 || n === 0) return 0;
  const dp = Array.from({ length: m + 1 }, () => new Uint16Array(n + 1));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
  return (1 - dp[m][n] / Math.max(m, n)) * 100;
}

export class Matcher {
  /**
   * @param {object} db - sql.js database
   * @param {object} vocabMap - { tconst_to_idx: {...}, n_items: N }
   * @param {object} akasEs - { normalized_spanish_title: [tconst, ...] }
   */
  constructor(db, vocabMap, akasEs) {
    this.db = db;
    this.vocabMap = vocabMap;
    this.akasEs = akasEs || {};

    // Build catalog lookup: normalized title -> [{tconst, year, votes}, ...]
    this.catalogByTitle = new Map();
    const rows = db.exec(
      'SELECT tconst, primary_title, original_title, start_year, num_votes FROM items'
    );
    if (rows.length) {
      for (const [tc, pt, ot, year, votes] of rows[0].values) {
        for (const title of [pt, ot]) {
          if (!title) continue;
          const key = normalize(title);
          if (!this.catalogByTitle.has(key)) this.catalogByTitle.set(key, []);
          this.catalogByTitle.get(key).push({ tconst: tc, year, votes: votes || 0 });
        }
      }
    }

    // Build akas lookup: normalized title -> [{tconst, year, votes}, ...]
    this.akasByTitle = new Map();
    for (const [normTitle, tconsts] of Object.entries(this.akasEs)) {
      const entries = [];
      for (const tc of tconsts) {
        const catalogRows = db.exec(
          'SELECT start_year, num_votes FROM items WHERE tconst = ? LIMIT 1', [tc]
        );
        if (catalogRows.length && catalogRows[0].values.length) {
          const [year, votes] = catalogRows[0].values[0];
          entries.push({ tconst: tc, year, votes: votes || 0 });
        }
      }
      if (entries.length) {
        this.akasByTitle.set(normTitle, entries);
      }
    }

    // Fuzzy pool: all known titles (catalog + akas)
    this.fuzzyPool = new Map();
    for (const [k, v] of this.catalogByTitle) this.fuzzyPool.set(k, v);
    for (const [k, v] of this.akasByTitle) {
      if (!this.fuzzyPool.has(k)) this.fuzzyPool.set(k, v);
      else this.fuzzyPool.get(k).push(...v);
    }
    this.fuzzyKeys = [...this.fuzzyPool.keys()];
  }

  _pick(candidates, year) {
    if (!candidates || !candidates.length) return null;
    let filtered = candidates;
    if (year != null) {
      filtered = candidates.filter(c => c.year && Math.abs(c.year - year) <= 1);
      if (!filtered.length) return null;
    }
    if (filtered.length === 1) return filtered[0].tconst;
    return filtered.sort((a, b) => b.votes - a.votes)[0].tconst;
  }

  _matchOne(title, year) {
    const norm = normalize(title);

    // Level 1: exact match in catalog (primary_title / original_title)
    let tc = this._pick(this.catalogByTitle.get(norm), year);
    if (tc) return tc;

    // Level 2: exact match in Spanish akas
    tc = this._pick(this.akasByTitle.get(norm), year);
    if (tc) return tc;

    // Level 3: fuzzy match (Levenshtein ratio >= 90)
    let bestKey = null, bestScore = 0;
    const threshold = year != null ? 90 : 94;
    for (const key of this.fuzzyKeys) {
      const score = levenshteinRatio(norm, key);
      if (score > bestScore) {
        bestScore = score;
        bestKey = key;
      }
    }
    if (bestKey && bestScore >= threshold) {
      tc = this._pick(this.fuzzyPool.get(bestKey), year);
      if (tc) return tc;
    }

    return null;
  }

  matchAll(events, onProgress) {
    const idxMap = this.vocabMap.tconst_to_idx || {};
    const sorted = [...events].sort((a, b) => {
      if (!a.timestamp && !b.timestamp) return 0;
      if (!a.timestamp) return -1;
      if (!b.timestamp) return 1;
      return a.timestamp - b.timestamp;
    });

    const sequence = [];
    const seenSet = new Set();
    const unmatched = [];
    let matched = 0;

    for (let i = 0; i < sorted.length; i++) {
      if (i % 20 === 0) onProgress?.(Math.round(i / sorted.length * 100));
      const ev = sorted[i];
      const tconst = this._matchOne(ev.title, ev.year);
      if (!tconst) {
        unmatched.push(`${ev.title} (${ev.year})`);
        continue;
      }
      matched++;
      const idx = idxMap[tconst];
      if (idx !== undefined && idx > 0) {
        sequence.push(idx);
        seenSet.add(idx);
      }
    }
    onProgress?.(100);
    return { sequence, seenSet, matched, total: events.length, unmatched };
  }
}
