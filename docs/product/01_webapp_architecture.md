# Block 1 — Webapp mínima: arquitectura client-side

> Fecha: 2026-07-09 · Status: BORRADOR · CLI: **DECIDIDO** → `frame-language-lm`

## Objetivo

Página web pública donde el usuario sube su perfil de FilmAffinity (export RGPD HTML) y recibe sus huecos — sin instalación, sin backend, coste de operación ~0.

## Inventario de artefactos a servir

| Artefacto | Tamaño | Necesario para |
|-----------|--------|----------------|
| model_fp32.onnx + .data (warm) | ~56 MB | Forward pass cine |
| model_full_fp32.onnx + .data (full) | ~101 MB | Forward pass series/fríos |
| item_embeddings.npy (warm) | 53 MB | Ranking cine |
| item_embeddings_full.npy (full) | 98 MB | Ranking series/fríos |
| full_aux.npz | 0.2 MB | Flags F2/F3 |
| catalog.sqlite | 36 MB | Matching + metadata display |
| vocab_map.json | 1 MB | ID↔título |
| feature_vocabs.json | 2.4 MB | Features composicionales |
| **Total (ambos paths)** | **~348 MB** | |
| **Total (solo warm)** | **~149 MB** | Solo recomendaciones de cine |

### Escenario reducido: solo cine (warm)

Si prescindimos del path frío (series/docs) en la webapp v1, los artefactos bajan a ~149 MB. Las series son el eslabón más débil del modelo (solo metadata, sin señal colaborativa, vecinos flojos) — no exponerlas en la webapp v1 es coherente con la decisión de HANDOFF de no exponer vecinos de series en producto.

**Recomendación:** webapp v1 solo con path warm (~149 MB). Path full como upgrade cuando haya más señal de series.

## Viabilidad de onnxruntime-web en navegador

### Estado de la tecnología (2026)

- **onnxruntime-web** soporta WASM (CPU) y WebGPU backends
- El backend WASM es maduro y funciona en todos los navegadores modernos (Chrome, Firefox, Safari, Edge)
- Modelos fp32 funcionan sin problemas — no hay degradación como con int8 dinámico
- **Limitación de tamaño:** los navegadores no tienen problemas cargando modelos de 50-100 MB en WASM, pero el tiempo de descarga inicial es el cuello de botella

### Forward pass: estimaciones

El modelo es diminuto (2 capas, d=256, ~5M parámetros):
- **CPU nativo (ONNX Runtime):** ~5.5 ms (medido en STATUS)
- **WASM en navegador:** estimado 3-10x más lento → **~15-55 ms** — imperceptible para el usuario
- **Matmul para ranking** (secuencia→embeddings): 54k×256 fp32 → ~50-200 ms en WASM

**Veredicto:** el forward pass en navegador es viable y rápido. El cuello de botella es la descarga inicial de artefactos, no la inferencia.

### Matching título→catálogo

`catalog.sqlite` se descarga directamente desde HF Hub (decisión de Pepe: incluido para conveniencia). Opciones para usarlo en el navegador:

1. **SQLite en navegador** (sql.js / wa-sqlite): cargar catalog.sqlite (~36 MB) en memoria WASM, buscar por título+año. Matching exacto por título+año cubre ~97% de casos (validado con import real de Pepe); fuzzy fallback con fuse.js para el resto
2. **Exportar JSON desde catalog.sqlite**: pre-procesar un `matching_index.json` (~5-10 MB gzipped) con `{titulo_es: id, titulo_original: id, año: X}` para matching rápido + un `catalog_display.json` para metadata de visualización

**Recomendación:** opción 1 — sql.js cargando catalog.sqlite directamente desde HF CDN. Es la más simple (un solo artefacto, sin paso de build), y sql.js es maduro en WASM. El 97% de matching por título+año hace innecesario un motor fuzzy pesado.

## Arquitectura propuesta

```
┌──────────────────────────────────────────────────────┐
│ HOSTING ESTÁTICO (GitHub Pages / HF Spaces static)   │
│                                                      │
│  index.html + app.js + styles.css                    │
│  (SPA ~500KB)                                        │
└──────────────┬───────────────────────────────────────┘
               │ fetch (lazy, on demand)
               ▼
┌──────────────────────────────────────────────────────┐
│ CDN DE ARTEFACTOS (HuggingFace Hub)                  │
│                                                      │
│  model_fp32.onnx.data         56 MB                  │
│  item_embeddings.npy          53 MB                  │
│  catalog.sqlite               36 MB                  │
│  vocab_map.json               1 MB                   │
│  feature_vocabs.json          2.4 MB                 │
│  (Total warm: ~149 MB)                               │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ NAVEGADOR DEL USUARIO                                │
│                                                      │
│  1. Sube movie-ratings.html (FA export)              │
│  2. Parser HTML → lista de (título, año, nota)       │
│  3. Matching contra catalog.sqlite (sql.js) → IDs    │
│  4. Forward pass ONNX (WASM) → distribución          │
│  5. Ranking: top-k \ vistas → huecos                │
│  6. Display con metadata de catalog.sqlite           │
│                                                      │
│  Todo client-side. Cero datos al servidor.           │
└──────────────────────────────────────────────────────┘
```

## Hosting: opciones

### GitHub Pages (recomendado para la app)

- Gratis, sin límite de tráfico razonable
- Dominio: `pepe.github.io/FrameLanguageLM` o custom
- Límite: 1 GB por repo → los artefactos NO caben aquí
- Ideal para: el código de la SPA (HTML/JS/CSS)

### HuggingFace Hub (recomendado para artefactos)

- Gratis para repos públicos, sin límite de tamaño práctico para ~300 MB
- Git LFS nativo — diseñado para servir archivos grandes
- CDN global (Cloudflare)
- URL estable: `https://huggingface.co/pepe/frame-language-lm/resolve/main/model_fp32.onnx.data`
- **Ideal para: los artefactos pesados (modelo, embeddings, índices)**

### HuggingFace Spaces (alternativa todo-en-uno)

- Permite hostear una app estática (HTML/JS) + artefactos en el mismo repo
- Gratis en tier CPU (sin backend, solo sirve estáticos)
- Dominio: `pepe-frame-language-lm.hf.space`
- Ventaja: un solo lugar para todo
- Desventaja: menos control que GitHub Pages, URL menos "profesional"

**Recomendación:** GitHub Pages para la SPA + HuggingFace Hub para artefactos. La SPA hace fetch de los artefactos desde HF Hub al cargar. Alternativa más simple: todo en HF Spaces.

## UX del flujo import → huecos

### Pantalla 1: Bienvenida
- Qué es FrameLanguageLM (1 párrafo)
- "Sube tu export de FilmAffinity para descubrir tus huecos"
- Tutorial visual: cómo pedir el export RGPD en FA (3-4 capturas)
- Botón: "Tengo mi export, empezar"

### Pantalla 2: Import
- Dropzone para el HTML (`movie-ratings.html` del ZIP RGPD)
- El fichero se procesa LOCALMENTE (mostrar "tus datos no salen de tu navegador")
- Barra de progreso: parsing → matching → listo
- Resumen: "Hemos reconocido X de Y películas (Z%)"
- Lista colapsable de no-matcheados

### Pantalla 3: Descarga del modelo (primera vez)
- "Descargando modelo y catálogo (~149 MB, solo la primera vez)"
- Barra de progreso
- Cachear en IndexedDB o Cache API para visitas futuras

### Pantalla 4: Huecos
- Lista de huecos ordenada por score, con:
  - Poster (de TMDB, con atribución)
  - Título (original + ES)
  - Año, país, director, género
  - Score de afinidad (percentil)
- Filtros: género, década, país (ver Block 3)
- Opción: "¿Merece la pena [título]?" (scoring puntual)

### Pantalla 5 (opcional): Worth
- El usuario escribe un título → score + explicación por features

## Consideraciones técnicas

### Caché de artefactos
- Usar Cache API (Service Worker) o IndexedDB para cachear los artefactos tras la primera descarga
- ~149 MB en caché es aceptable para navegadores modernos
- Segundo uso: carga instantánea, sin red

### Compatibilidad
- onnxruntime-web WASM funciona en Chrome 87+, Firefox 89+, Safari 15+, Edge 87+
- Cobertura estimada: >95% de usuarios
- Fallback para navegadores viejos: no (la VISION dice "usuario medio", no "todos")

### Privacidad
- Cero datos enviados al servidor — todo client-side
- El fichero FA se procesa en memoria y se descarta
- Los artefactos se descargan de HF Hub (solo lectura, sin tracking)
- Mostrar "Privacy-first: your data never leaves your browser" de forma prominente

## Stack sugerido

- **Framework:** vanilla JS o Preact (~3KB) — no React/Vue/Angular para minimizar bundle
- **ONNX:** onnxruntime-web (WASM backend)
- **SQLite en browser:** sql.js (~1MB WASM) para cargar catalog.sqlite desde HF CDN
- **Matching:** búsqueda exacta título+año vía SQL (97% hit rate), fuse.js (~25KB) como fallback fuzzy
- **UI:** CSS moderno (grid/flexbox), sin framework CSS

## Coste de operación

| Concepto | Coste |
|----------|-------|
| GitHub Pages | Gratis |
| HuggingFace Hub | Gratis |
| Dominio custom (opcional) | ~10€/año |
| **Total** | **~0€/año** |

Alineado con la línea roja "prácticamente gratuito de operar".

## Decisiones de implementación (abiertas, no bloquean borradores)

1. ¿Webapp v1 solo cine (warm, ~149 MB) o incluir series (full, ~348 MB)?
2. ¿GitHub Pages + HF Hub o todo en HF Spaces?
3. ¿Incluir posters de TMDB? (requiere atribución TMDB — estrategia de buena fe decidida, pero posters en tiempo real vs cacheados es otra cuestión)
4. ¿Vanilla JS o Preact?
5. ¿Dominio custom?

> Nota: estas son decisiones de implementación para cuando se construya la webapp, no bloquean la publicación de borradores.
