/**
 * Post-hoc filter logic over recommendation results using catalog metadata.
 */

export class FilterEngine {
  constructor(db) {
    this.db = db;
    this._metaCache = new Map();
    this._availableGenres = null;
    this._availableCountries = null;
    this._availableLanguages = null;
  }

  getMeta(tconst) {
    if (this._metaCache.has(tconst)) return this._metaCache.get(tconst);
    const rows = this.db.exec(
      `SELECT primary_title, original_title, start_year, title_type,
              genres, directors, countries, original_language, poster_path, num_votes
       FROM items WHERE tconst = ? LIMIT 1`, [tconst]
    );
    if (!rows.length || !rows[0].values.length) return null;
    const [pt, ot, year, tt, genres, dirs, countries, lang, poster, votes] = rows[0].values[0];
    const meta = {
      tconst,
      title: pt || ot,
      originalTitle: ot,
      year,
      type: tt,
      genres: (genres || '').split(',').map(g => g.trim()).filter(Boolean),
      directors: (dirs || '').split('|').map(d => d.trim()).filter(Boolean),
      countries: (countries || '').split('|').map(c => c.trim()).filter(Boolean),
      language: lang,
      poster: poster,
      votes: votes || 0,
    };
    this._metaCache.set(tconst, meta);
    return meta;
  }

  getAvailableGenres() {
    if (this._availableGenres) return this._availableGenres;
    const rows = this.db.exec('SELECT DISTINCT genres FROM items WHERE genres IS NOT NULL');
    const set = new Set();
    if (rows.length) {
      for (const [g] of rows[0].values) {
        for (const genre of (g || '').split(',')) {
          const trimmed = genre.trim();
          if (trimmed) set.add(trimmed);
        }
      }
    }
    this._availableGenres = [...set].sort();
    return this._availableGenres;
  }

  getAvailableCountries() {
    if (this._availableCountries) return this._availableCountries;
    const rows = this.db.exec('SELECT DISTINCT countries FROM items WHERE countries IS NOT NULL');
    const set = new Set();
    if (rows.length) {
      for (const [c] of rows[0].values) {
        for (const country of (c || '').split('|')) {
          const trimmed = country.trim();
          if (trimmed) set.add(trimmed);
        }
      }
    }
    this._availableCountries = [...set].sort();
    return this._availableCountries;
  }

  getAvailableLanguages() {
    if (this._availableLanguages) return this._availableLanguages;
    const rows = this.db.exec(
      'SELECT DISTINCT original_language FROM items WHERE original_language IS NOT NULL'
    );
    const set = new Set();
    if (rows.length) {
      for (const [l] of rows[0].values) if (l) set.add(l);
    }
    this._availableLanguages = [...set].sort();
    return this._availableLanguages;
  }

  applyFilters(items, filters) {
    return items.filter(item => {
      const meta = item.meta;
      if (!meta) return false;

      if (filters.type && meta.type !== filters.type) return false;

      if (filters.genres?.length) {
        const match = filters.genres.some(g => meta.genres.includes(g));
        if (!match) return false;
      }

      if (filters.decadeStart || filters.decadeEnd) {
        const y = meta.year;
        if (filters.decadeStart && y < filters.decadeStart) return false;
        if (filters.decadeEnd && y > filters.decadeEnd) return false;
      }

      if (filters.countries?.length) {
        const match = filters.countries.some(c => meta.countries.includes(c));
        if (!match) return false;
      }

      if (filters.language && meta.language !== filters.language) return false;

      if (filters.director) {
        const norm = filters.director.toLowerCase();
        const match = meta.directors.some(d => d.toLowerCase().includes(norm));
        if (!match) return false;
      }

      return true;
    });
  }
}
