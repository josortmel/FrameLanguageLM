/**
 * FrameLanguageLM webapp — main application controller.
 */
import { parseFilmAffinityZip } from './parser.js';
import { Matcher } from './matching.js';
import { Recommender } from './inference.js';
import { FilterEngine } from './filters.js';
import {
  showScreen, updateProgress, showMatchStats,
  renderResults, renderGrid, buildFilterUI, updateFilterCount,
} from './ui.js';

const CONFIG = {
  ARTIFACTS_BASE: 'https://huggingface.co/josortmel/FrameLanguageLM/resolve/main',
};

let state = {
  db: null,
  vocabMap: null,
  recommender: null,
  filterEngine: null,
  sequence: null,
  seenSet: null,
  allResults: [],
  matchStats: null,
  viewMode: 'grid',
};

async function loadCatalog() {
  updateProgress('Loading catalog...', 10);
  const [sqlPromise, vocabResp] = await Promise.all([
    initSqlJs({ locateFile: f => `https://sql.js.org/dist/${f}` }),
    fetch(`${CONFIG.ARTIFACTS_BASE}/vocab_map.json`),
  ]);
  const SQL = sqlPromise;
  const catalogResp = await fetch(`${CONFIG.ARTIFACTS_BASE}/catalog.sqlite`);
  const catalogBuf = await catalogResp.arrayBuffer();
  state.db = new SQL.Database(new Uint8Array(catalogBuf));
  state.vocabMap = await vocabResp.json();
  state.filterEngine = new FilterEngine(state.db);
  updateProgress('Catalog loaded', 30);
}

async function loadModel() {
  updateProgress('Loading model (this may take a moment)...', 40);
  state.recommender = new Recommender();
  await state.recommender.load(
    `${CONFIG.ARTIFACTS_BASE}/model_fp32.onnx`,
    `${CONFIG.ARTIFACTS_BASE}/item_embeddings.npy`,
    msg => updateProgress(msg, 60)
  );
  updateProgress('Model ready', 80);
}

async function handleFileUpload(file) {
  showScreen('screen-progress');
  updateProgress('Parsing FilmAffinity export...', 5);

  try {
    const events = await parseFilmAffinityZip(file, pct =>
      updateProgress(`Parsing ratings... ${pct}%`, 5 + pct * 0.15)
    );
    updateProgress(`Parsed ${events.length} ratings. Loading catalog...`, 20);

    await loadCatalog();

    updateProgress('Loading Spanish title mappings...', 33);
    const akasResp = await fetch(`${CONFIG.ARTIFACTS_BASE}/akas_es.json`);
    const akasEs = await akasResp.json();

    updateProgress('Matching titles to catalog...', 35);
    const matcher = new Matcher(state.db, state.vocabMap, akasEs);
    const result = matcher.matchAll(events, pct =>
      updateProgress(`Matching... ${pct}%`, 35 + pct * 0.15)
    );
    state.sequence = result.sequence;
    state.seenSet = result.seenSet;
    state.matchStats = {
      ...result,
      inVocab: result.sequence.length,
    };

    await loadModel();

    updateProgress('Computing recommendations...', 85);
    const raw = await state.recommender.predict(
      state.sequence, state.seenSet, 200
    );

    const idxToTconst = {};
    if (state.vocabMap.tconst_to_idx) {
      for (const [tc, idx] of Object.entries(state.vocabMap.tconst_to_idx)) {
        idxToTconst[idx] = tc;
      }
    }
    state.allResults = raw.map(r => {
      const tconst = idxToTconst[r.index];
      return {
        ...r,
        tconst,
        meta: tconst ? state.filterEngine.getMeta(tconst) : null,
      };
    }).filter(r => r.meta);

    updateProgress('Done!', 100);
    showResults();
  } catch (err) {
    updateProgress(`Error: ${err.message}`, 0);
    console.error(err);
  }
}

function renderCurrentView(items, container) {
  const display = items.slice(0, 50);
  if (state.viewMode === 'grid') {
    renderGrid(display, container);
  } else {
    renderResults(display, container);
  }
  updateFilterCount(display.length);
}

function showResults() {
  showScreen('screen-results');
  showMatchStats(state.matchStats);

  const container = document.getElementById('results-list');
  let currentFilters = {};

  renderCurrentView(state.allResults, container);

  buildFilterUI(state.filterEngine, filters => {
    currentFilters = filters;
    const filtered = state.filterEngine.applyFilters(state.allResults, filters);
    renderCurrentView(filtered, container);
  });

  document.getElementById('view-list')?.addEventListener('click', () => {
    state.viewMode = 'list';
    document.getElementById('view-list').classList.add('active');
    document.getElementById('view-grid').classList.remove('active');
    const filtered = Object.keys(currentFilters).length
      ? state.filterEngine.applyFilters(state.allResults, currentFilters)
      : state.allResults;
    renderCurrentView(filtered, container);
  });

  document.getElementById('view-grid')?.addEventListener('click', () => {
    state.viewMode = 'grid';
    document.getElementById('view-grid').classList.add('active');
    document.getElementById('view-list').classList.remove('active');
    const filtered = Object.keys(currentFilters).length
      ? state.filterEngine.applyFilters(state.allResults, currentFilters)
      : state.allResults;
    renderCurrentView(filtered, container);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  showScreen('screen-welcome');

  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');

  if (dropzone) {
    dropzone.addEventListener('dragover', e => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });
    dropzone.addEventListener('drop', e => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      const file = e.dataTransfer.files[0];
      if (file) handleFileUpload(file);
    });
    dropzone.addEventListener('click', () => fileInput?.click());
  }

  if (fileInput) {
    fileInput.addEventListener('change', e => {
      const file = e.target.files[0];
      if (file) handleFileUpload(file);
    });
  }

  document.getElementById('btn-start')?.addEventListener('click', () => {
    showScreen('screen-upload');
  });
});
