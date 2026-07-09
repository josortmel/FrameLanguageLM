/**
 * Parse FilmAffinity GDPR export ZIP → array of rating events.
 * Port of frame_language_lm/importers/filmaffinity.py
 */

const MESES = {
  enero: 1, febrero: 2, marzo: 3, abril: 4, mayo: 5, junio: 6,
  julio: 7, agosto: 8, septiembre: 9, octubre: 10, noviembre: 11, diciembre: 12,
};

const ROW_RE = /<div class="user-rating">(\d+)<\/div><\/td>\s*<td>(.*?)<\/td>\s*<td><em>(.*?)<\/em>/gs;
const TITLE_YEAR_RE = /^(.*)\s+\((\d{4})\)\s*$/s;
const DATE_RE = /(\d{1,2}) de (\w+) de (\d{4}),?\s*(\d{1,2}):(\d{2})/;

function parseDate(text) {
  const m = DATE_RE.exec(text);
  if (!m) return null;
  const [, d, mes, y, hh, mm] = m;
  const month = MESES[mes.toLowerCase()];
  if (!month) return null;
  return new Date(+y, month - 1, +d, +hh, +mm);
}

function decodeEntities(str) {
  const el = document.createElement('textarea');
  el.innerHTML = str;
  return el.value;
}

/**
 * @param {File} file - ZIP file from user
 * @param {function} onProgress - callback(pct)
 * @returns {Promise<{title: string, year: number|null, rating: number, timestamp: Date|null, source: string}[]>}
 */
export async function parseFilmAffinityZip(file, onProgress) {
  onProgress?.(0);
  const zip = await JSZip.loadAsync(file);
  const htmlName = Object.keys(zip.files).find(n => n.endsWith('movie-ratings.html'));
  if (!htmlName) throw new Error('movie-ratings.html not found in ZIP');
  const raw = await zip.file(htmlName).async('string');
  onProgress?.(30);

  const events = [];
  let match;
  ROW_RE.lastIndex = 0;
  while ((match = ROW_RE.exec(raw)) !== null) {
    const [, rating10, tdTitle, tdDate] = match;
    const text = decodeEntities(tdTitle).trim();
    const tm = TITLE_YEAR_RE.exec(text);
    const title = tm ? tm[1].trim() : text;
    const year = tm ? parseInt(tm[2], 10) : null;
    events.push({
      title,
      year,
      rating: parseInt(rating10, 10) / 2.0,
      timestamp: parseDate(decodeEntities(tdDate)),
      source: 'filmaffinity',
    });
  }
  onProgress?.(100);
  return events;
}
