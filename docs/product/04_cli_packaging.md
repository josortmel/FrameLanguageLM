# Block 4 — CLI producto: empaquetado usable

> Fecha: 2026-07-09 · Status: BORRADOR · Nombre CLI: **DECIDIDO** → `frame-language-lm`

## Estado actual

Los scripts de desarrollo (`scripts/import_user.py`, `scripts/dual_report.py`) funcionan pero son de desarrollador:
- Rutas hardcodeadas o relativas al repo
- Requieren el repo clonado + `uv` + dependencias
- Sin ayuda inline, sin manejo de errores de usuario

## Propuesta Fase 6 del PLAN: comandos

```bash
# Import: el usuario trae sus datos
frame-language-lm import --filmaffinity export.zip
frame-language-lm import --netflix ViewingActivity.csv --profile "Jose Antonio"
frame-language-lm import --letterboxd export.zip
frame-language-lm import --imdb ratings.csv
frame-language-lm import --trakt username

# Huecos: la pieza central
frame-language-lm gaps [--top 50] [--genre X] [--decade 1990s] [--country FR] [--type movie]

# Similar: vecinos de un título
frame-language-lm similar "Mulholland Drive" [--country FR]

# Worth: ¿me merece la pena?
frame-language-lm worth "Twin Peaks"

# Why: explica la recomendación
frame-language-lm why "Oldboy"

# Search: búsqueda paramétrica
frame-language-lm search --director "Park Chan-wook" --language ko
```

## Nombre del CLI: DECIDIDO

**`frame-language-lm`** — comando y paquete pip. El paquete Python interno sigue siendo `framelm` por ahora (se renombrará más adelante, fuera del alcance de producto).

## Instalación

```bash
# Opción 1: uv (recomendada para early adopters)
uv tool install frame-language-lm

# Opción 2: pip
pip install frame-language-lm

# Opción 3: pipx
pipx install frame-language-lm
```

### Primera ejecución: descarga automática

Al ejecutar cualquier comando por primera vez, el CLI debe:
1. Crear `~/.frame-language-lm/`
2. Descargar artefactos de HF Hub (~348 MB: modelo + embeddings + catálogo)
3. Mostrar progreso y tiempo estimado

El catálogo (`catalog.sqlite`) viene pre-construido desde HF — no requiere API keys ni pasos adicionales del usuario. `frame-language-lm setup` existe como comando explícito opcional (para forzar re-descarga o actualización), pero no es necesario: la primera ejecución de cualquier comando lo hace automáticamente.

Para ejecuciones posteriores, los artefactos se cachean localmente.

## Estructura del paquete Python

```
framelm/                             # paquete interno (renombrar pendiente)
├── __init__.py
├── __main__.py          # Entry point: python -m framelm
├── cli.py               # Argparse / Click commands
├── inference.py          # ONNX forward pass + ranking
├── matching.py           # Título → ID catálogo (fuzzy)
├── importers/
│   ├── filmaffinity.py   # Parser HTML RGPD
│   ├── netflix.py        # Parser ViewingActivity.csv
│   ├── letterboxd.py     # Parser ZIP
│   ├── imdb.py           # Parser ratings CSV
│   └── trakt.py          # API client
├── catalog.py            # SQLite queries + filtros
├── explain.py            # why/worth: explicación por features
└── display.py            # Formato de salida (tabla, colores)
```

`pyproject.toml`:
```toml
[project]
name = "frame-language-lm"
version = "1.0.0"
requires-python = ">=3.10"
dependencies = [
    "onnxruntime>=1.18",
    "numpy>=1.24",
    "rapidfuzz>=3.0",
    "rich>=13.0",       # tablas bonitas en terminal
    "huggingface-hub",  # descarga de artefactos
]

[project.scripts]
frame-language-lm = "framelm.cli:main"
```

## Output: formato de huecos

```
 Tu perfil: 834 títulos reconocidos (96.5% matching)

 Top 50 huecos (cine):

  #  Título                        Año   Género              País  Director          Afinidad
  1  In the Mood for Love           2000  Drama, Romance      HK    Wong Kar-wai      Top 1%
  2  Stalker                        1979  Drama, Sci-Fi       SU    Andrei Tarkovsky  Top 1%
  3  Yi Yi                          2000  Drama               TW    Edward Yang       Top 2%
  ...

 Series (cold-start, calidad experimental):

  #  Título                        Año   Género              País  Afinidad
  1  Shogun                         2024  Drama, History      US    Top 3%
  2  The Bear                       2022  Comedy, Drama       US    Top 4%
  ...
```

Usar `rich` para tablas con colores en terminal (degradado del score con color).

## Datos del usuario: privacidad

- El perfil del usuario se guarda en `~/.frame-language-lm/profiles/`
- Formato: parquet (eficiente, estándar)
- **Nunca se transmite a ningún servidor** — toda la inferencia es local
- El usuario puede borrar su perfil con `frame-language-lm profile delete`

## Decisiones — todas DECIDIDAS

1. ~~**Nombre del CLI**~~ → **`frame-language-lm`**
2. ~~**Setup automático vs explícito**~~ → **Automático en primera ejecución** (setup explícito queda como opcional)
3. ~~**API key de TMDB para catálogo**~~ → **No necesaria** — catálogo pre-construido incluido en HF

## Decisiones de implementación (abiertas, no bloquean borradores)

1. **¿Incluir `rich` como dependencia?** — añade color y tablas bonitas pero es una dependencia más

> Nota: decisión menor de implementación, para cuando se construya el CLI.
