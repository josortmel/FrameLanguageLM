# Block 3 — Filtros de búsqueda para el usuario

> Fecha: 2026-07-09 · Status: BORRADOR · CLI: **DECIDIDO** → `frame-language-lm`

## Contexto

Caso de uso 2 del SPEC: búsqueda paramétrica. El usuario no solo quiere "la siguiente película" sino "la siguiente comedia francesa de los 90". El catálogo (`catalog.sqlite`, 36 MB) ya tiene los campos necesarios.

## Campos disponibles en catalog.sqlite

| Campo | Fuente | Cardinalidad | Cobertura | Tipo de filtro |
|-------|--------|:------------:|:---------:|----------------|
| Género | IMDb `title.basics` | ~28 | ~99% | Multi-select (una película tiene varios géneros) |
| Década | IMDb `startYear` | ~13 (1920s-2020s) | ~99% | Select / slider |
| País | TMDB | ~150 | ~90% | Multi-select (coproducciones) |
| Idioma original | TMDB | ~100 | ~90% | Select |
| Director | IMDb `title.crew` | ~30k | ~95% | Autocomplete text |
| Tipo | IMDb `titleType` | 3 (movie/tvSeries/tvMiniSeries) | 100% | Select |
| Cast | IMDb `title.principals` | ~100k | ~95% | Autocomplete text |
| Presupuesto | TMDB (bucketizado) | 6-8 buckets | 20-40% | Select (con "desconocido") |

## Propuesta: CLI

```bash
# Filtros como flags sobre el comando gaps
frame-language-lm gaps --top 50 --genre "Thriller" --decade 1990s --country FR
frame-language-lm gaps --top 30 --director "Park Chan-wook" --language ko
frame-language-lm gaps --type tvSeries --genre "Sci-Fi"

# Búsqueda paramétrica pura (caso de uso 2)
frame-language-lm search --director "Denis Villeneuve" --min-year 2010 --language en
frame-language-lm search --genre "Animation" --country JP --decade 2000s

# Similar (vecinos en espacio de embeddings + filtros)
frame-language-lm similar "Mulholland Drive" --country FR --genre "Thriller"
```

Mecánica:
1. `gaps` con filtros: el forward pass genera la distribución completa, luego se aplica máscara SQL sobre el catálogo → solo se muestran huecos que cumplen los filtros
2. `search`: filtro SQL primero (reduce candidatos), luego ranking por similitud en el espacio de embeddings contra el centroide del historial del usuario (o un query-vector composicional si no hay historial)
3. `similar` con filtros: vecinos del ítem por coseno, filtrados post-hoc por SQL

Los filtros son post-hoc (no afectan al modelo), lo cual es correcto — el modelo genera la distribución sobre TODO el catálogo y los filtros solo seleccionan qué mostrar.

## Propuesta: Webapp

### Panel de filtros (sidebar o drawer colapsable)

```
┌─────────────────────────────────────────────┐
│ Filtros                              [Reset] │
│                                              │
│ Tipo:    [Cine ▼] [Series ▼] [Docs ▼]      │
│ Género:  [Drama ×] [Thriller ×] [+ Añadir]  │
│ Década:  [1960] ────●────── [2026]          │
│ País:    [_____________▼] (autocomplete)     │
│ Idioma:  [_____________▼]                    │
│ Director:[_____________] (autocomplete)      │
│                                              │
│ [Aplicar filtros]                            │
└─────────────────────────────────────────────┘
```

Mecánica en el navegador:
- Los filtros se aplican sobre el array de resultados del forward pass (ya en memoria)
- No hace falta reejecutar el modelo — solo filtrar y re-renderizar la lista
- Si usamos JSON en vez de SQLite, los filtros son simples `.filter()` en JS
- Instantáneo (<10 ms para 100k items)

### UX considerations

1. **Género multi-select:** las películas tienen varios géneros. Filtrar por "Thriller" debe incluir películas que son "Thriller, Drama" (OR, no AND)
2. **País multi-select:** coproducciones (FR/US → aparece si filtras por FR O por US)
3. **Década como slider:** más intuitivo que un dropdown para rangos
4. **Director/Cast como autocomplete:** con los ~30k/~100k opciones, un dropdown no funciona. Autocomplete con debounce sobre el JSON en memoria
5. **Presupuesto:** NO exponer como filtro público — 60-80% de "desconocido" lo hace inútil para el usuario. Solo relevante internamente como feature del modelo
6. **Conteo dinámico:** mostrar "X huecos encontrados" al aplicar filtros, para que el usuario vea si sus filtros son demasiado restrictivos

## Datos necesarios para filtros en la webapp

Si no podemos servir `catalog.sqlite` (problema de licencias), necesitamos un JSON de metadata:

```json
{
  "items": [
    {
      "id": 12345,
      "title": "Mulholland Drive",
      "title_es": "Mulholland Drive",
      "year": 2001,
      "type": "movie",
      "genres": ["Drama", "Mystery", "Thriller"],
      "country": ["US", "FR"],
      "language": "en",
      "director": ["David Lynch"],
      "cast": ["Naomi Watts", "Laura Harring"],
      "poster_path": "/tVxGt7ZtCim4JESaQFMO7dDGESo.jpg"
    }
  ]
}
```

Tamaño estimado (100k items, campos esenciales): ~15-20 MB sin comprimir, ~3-5 MB gzipped.

**Problema de licencias:** este JSON contiene datos de IMDb y TMDB. Mismo problema que catalog.sqlite (ver Block 5). Opciones:
1. Generar localmente con script (usuario construye su catálogo)
2. Usar solo campos de Wikidata/OpenData para la versión pública
3. La webapp obtiene los datos en tiempo real de la API de TMDB con la key del usuario (overkill)

**Recomendación:** la webapp incluye el JSON de metadata como artefacto estático en HF Hub. Los datos son factuales (títulos, años, géneros) — no creativos ni originales. El riesgo legal de redistribuir metadatos factuales es mínimo, especialmente bajo uso no-comercial. Incluir atribución a IMDb y TMDB.

## Decisiones de implementación (abiertas, no bloquean borradores)

1. ¿Exposición de presupuesto como filtro? (recomendación: no, por baja cobertura)
2. ¿Cast como filtro en webapp? (añade complejidad y peso al JSON — quizá solo en CLI)
3. ¿Slider de década o dropdown?

> Nota: decisiones menores de UX, para cuando se construyan el CLI y la webapp.
