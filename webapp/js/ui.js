/**
 * UI rendering: screens, results, filters.
 */

const GENRE_COLORS = {
  'Drama': '#5b7fa3', 'Comedy': '#e8a838', 'Action': '#c0392b',
  'Thriller': '#2c3e50', 'Horror': '#8e44ad', 'Romance': '#e74c8c',
  'Sci-Fi': '#2980b9', 'Animation': '#27ae60', 'Documentary': '#16a085',
  'Crime': '#7f8c8d', 'Adventure': '#d35400', 'Fantasy': '#8e44ad',
  'Mystery': '#34495e', 'War': '#6c3b2a', 'Music': '#e67e22',
  'Biography': '#1abc9c', 'History': '#795548', 'Western': '#a0522d',
  'Sport': '#3498db', 'Family': '#2ecc71', 'Musical': '#9b59b6',
};

function genreColor(genres) {
  for (const g of genres) {
    if (GENRE_COLORS[g]) return GENRE_COLORS[g];
  }
  return '#555';
}

function percentileLabel(pct) {
  if (pct >= 99) return { text: 'Top 1%', cls: 'pct-top1' };
  if (pct >= 95) return { text: `Top ${(100 - pct).toFixed(0)}%`, cls: 'pct-top5' };
  if (pct >= 90) return { text: `Top ${(100 - pct).toFixed(0)}%`, cls: 'pct-top10' };
  return { text: `${pct.toFixed(0)}%`, cls: 'pct-normal' };
}

export function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id)?.classList.add('active');
}

export function updateProgress(text, pct = null) {
  const el = document.getElementById('progress-text');
  const bar = document.getElementById('progress-bar-fill');
  if (el) el.textContent = text;
  if (bar && pct !== null) {
    bar.style.width = `${pct}%`;
  }
}

export function showMatchStats(stats) {
  const el = document.getElementById('match-stats');
  if (!el) return;
  const pct = ((stats.matched / stats.total) * 100).toFixed(1);
  el.innerHTML = `
    <div class="stat-row">
      <span class="stat-label">Ratings parsed</span>
      <span class="stat-value">${stats.total}</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">Matched to catalog</span>
      <span class="stat-value">${stats.matched} (${pct}%)</span>
    </div>
    <div class="stat-row">
      <span class="stat-label">In model vocabulary</span>
      <span class="stat-value">${stats.inVocab}</span>
    </div>
    ${stats.unmatched.length ? `
    <details class="unmatched-details">
      <summary>${stats.unmatched.length} unmatched titles</summary>
      <ul>${stats.unmatched.slice(0, 50).map(t => `<li>${t}</li>`).join('')}</ul>
      ${stats.unmatched.length > 50 ? `<p>...and ${stats.unmatched.length - 50} more</p>` : ''}
    </details>` : ''}
  `;
}

export function renderResults(items, container) {
  if (!container) return;
  if (!items.length) {
    container.innerHTML = '<p class="no-results">No results match your filters.</p>';
    return;
  }
  container.innerHTML = items.map((item, i) => {
    const m = item.meta;
    if (!m) return '';
    const pct = percentileLabel(item.percentile);
    const color = genreColor(m.genres);
    return `
    <div class="result-card">
      <div class="card-rank">${i + 1}</div>
      ${m.poster
        ? `<img class="card-poster" src="https://image.tmdb.org/t/p/w342${m.poster}" alt="${m.title}" loading="lazy" onerror="this.outerHTML='<div class=\\'card-poster\\' style=\\'background:${color}\\'><span class=\\'poster-letter\\'>${(m.title||'?')[0]}</span></div>'">`
        : `<div class="card-poster" style="background-color: ${color}"><span class="poster-letter">${(m.title || '?')[0]}</span></div>`
      }
      <div class="card-info">
        <div class="card-title">${m.title}</div>
        ${m.originalTitle && m.originalTitle !== m.title
          ? `<div class="card-original">${m.originalTitle}</div>` : ''}
        <div class="card-meta">
          <span class="card-year">${m.year || '?'}</span>
          <span class="card-dot">&middot;</span>
          <span class="card-genres">${m.genres.slice(0, 3).join(', ')}</span>
          ${m.countries.length ? `<span class="card-dot">&middot;</span><span class="card-country">${m.countries.join(', ')}</span>` : ''}
        </div>
        ${m.directors.length ? `<div class="card-director">${m.directors.join(', ')}</div>` : ''}
      </div>
      <div class="card-score ${pct.cls}">${pct.text}</div>
    </div>`;
  }).join('');
}

export function buildFilterUI(filterEngine, onApply) {
  const panel = document.getElementById('filter-panel');
  if (!panel) return;

  const genres = filterEngine.getAvailableGenres();
  const countries = filterEngine.getAvailableCountries();
  const languages = filterEngine.getAvailableLanguages();

  panel.innerHTML = `
    <h3>Filters</h3>
    <div class="filter-group">
      <label>Type</label>
      <select id="f-type">
        <option value="">All</option>
        <option value="movie">Movies</option>
        <option value="tvSeries">TV Series</option>
        <option value="tvMiniSeries">Mini Series</option>
      </select>
    </div>
    <div class="filter-group">
      <label>Genre</label>
      <select id="f-genre">
        <option value="">All</option>
        ${genres.map(g => `<option value="${g}">${g}</option>`).join('')}
      </select>
    </div>
    <div class="filter-group">
      <label>Decade</label>
      <div class="decade-range">
        <input type="number" id="f-year-from" placeholder="From" min="1920" max="2026" step="10">
        <span>&ndash;</span>
        <input type="number" id="f-year-to" placeholder="To" min="1920" max="2026" step="10">
      </div>
    </div>
    <div class="filter-group">
      <label>Country</label>
      <select id="f-country">
        <option value="">All</option>
        ${countries.map(c => `<option value="${c}">${c}</option>`).join('')}
      </select>
    </div>
    <div class="filter-group">
      <label>Language</label>
      <select id="f-language">
        <option value="">All</option>
        ${languages.map(l => `<option value="${l}">${l}</option>`).join('')}
      </select>
    </div>
    <div class="filter-group">
      <label>Director</label>
      <input type="text" id="f-director" placeholder="Search director...">
    </div>
    <div class="filter-actions">
      <button id="apply-filters" class="btn btn-accent">Apply</button>
      <button id="reset-filters" class="btn btn-secondary">Reset</button>
    </div>
    <div id="filter-count" class="filter-count"></div>
  `;

  const gather = () => ({
    type: document.getElementById('f-type').value || null,
    genres: document.getElementById('f-genre').value
      ? [document.getElementById('f-genre').value] : [],
    decadeStart: parseInt(document.getElementById('f-year-from').value) || null,
    decadeEnd: parseInt(document.getElementById('f-year-to').value) || null,
    countries: document.getElementById('f-country').value
      ? [document.getElementById('f-country').value] : [],
    language: document.getElementById('f-language').value || null,
    director: document.getElementById('f-director').value || null,
  });

  document.getElementById('apply-filters').addEventListener('click', () => onApply(gather()));
  document.getElementById('reset-filters').addEventListener('click', () => {
    panel.querySelectorAll('select').forEach(s => s.value = '');
    panel.querySelectorAll('input').forEach(i => i.value = '');
    onApply({});
  });
}

export function renderGrid(items, container) {
  if (!container) return;
  if (!items.length) {
    container.innerHTML = '<p class="no-results">No results match your filters.</p>';
    return;
  }
  container.innerHTML = `<div class="poster-grid">${items.map((item, i) => {
    const m = item.meta;
    if (!m) return '';
    const pct = percentileLabel(item.percentile);
    const color = genreColor(m.genres);
    return `
    <div class="grid-card">
      <div class="grid-rank">${i + 1}</div>
      ${m.poster
        ? `<img src="https://image.tmdb.org/t/p/w342${m.poster}" alt="${m.title}" loading="lazy" onerror="this.outerHTML='<div class=\\'grid-fallback\\' style=\\'background:${color}\\'>${(m.title||'?')[0]}</div>'">`
        : `<div class="grid-fallback" style="background:${color}">${(m.title || '?')[0]}</div>`
      }
      <div class="grid-score ${pct.cls}">${pct.text}</div>
      <div class="grid-overlay">
        <div class="grid-title">${m.title}</div>
        <div class="grid-meta">${m.year || '?'} · ${m.genres.slice(0, 2).join(', ')}</div>
      </div>
    </div>`;
  }).join('')}</div>`;
}

export function updateFilterCount(n) {
  const el = document.getElementById('filter-count');
  if (el) el.textContent = `${n} result${n !== 1 ? 's' : ''}`;
}
