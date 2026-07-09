# Block 2 — FilmAffinity Public Profile: viabilidad de lectura directa

> Fecha: 2026-07-09 · Status: BORRADOR de investigación · Scraping FA: **descartado** (ver hallazgos abajo)

## Pregunta

Cuando el perfil FA del usuario es público, ¿podemos leer sus votos directamente de la web sin export RGPD?

## Hallazgos

### 1. robots.txt (verificado 2026-07-09)

```
User-agent: *
Disallow: /*?FASID
Disallow: /*&FASID
Disallow: /*/sharerating
Disallow: /*/userratings
Disallow: /*/userrating
Disallow: /*/names-together
```

**Las URLs de ratings de usuario (`/*/userratings`, `/*/userrating`) están explícitamente prohibidas para crawlers.** Esto incluye cualquier variante de perfil público. FA no quiere que los bots accedan a los votos de los usuarios.

### 2. Terms of Use / Privacy Policy (verificado 2026-07-09)

Del texto de `private.php` (la página legal visible):

> "Any sign of inappropriate use of the website, or tampering thereof will result in the subsequent deletion of suspicious accounts."

> "FilmAffinity has implemented a series of security protocols [...] bots, automated voting [...] are no longer considered in the average ratings."

> "Under no circumstances will this information be shared with a third party."

No existe un apartado explícito de "Terms of Service" con cláusulas sobre scraping tipo Spotify/Goodreads. Pero las señales son claras:
- robots.txt prohíbe `/*/userratings` explícitamente
- Detección activa de bots
- El servidor devuelve **403 Forbidden** a user-agents automatizados estándar (verificado: WebFetch bloqueado)
- No hay API pública

### 3. Comparación con precedentes

| Plataforma | API pública | robots.txt ratings | ToS scraping | Estado |
|------------|:-----------:|:------------------:|:------------:|--------|
| Goodreads | Cerrada (2024) | — | Prohibido | Muerta |
| Spotify | No para ML | — | "may not [...] for training ML" | Prohibido |
| Letterboxd | No oficial | Permisivo | Gris | Solo export |
| **FilmAffinity** | **No** | **Prohibe userratings** | **Sin API, bots detectados, 403** | **Hostil** |

### 4. Viabilidad técnica (si fuera legal)

**Client-side (navegador del usuario):** imposible por CORS. FA no envía `Access-Control-Allow-Origin` en sus respuestas. El fetch desde un dominio distinto falla. Solo funcionaría si el propio usuario instalase una extensión de navegador (que inyecta en la página de FA), pero eso requiere instalación → viola la línea roja "cero instalación para el usuario medio".

**Server-side (proxy):** técnicamente viable con headers/cookies adecuados (el stealthy_fetch con Playwright lo logra), pero:
- Viola robots.txt explícito
- Cada petición sale de NUESTRO servidor → FA puede bloquear IP, rate-limit, o tomar acciones legales
- Introduce coste de servidor → viola la línea roja "coste operación ~0"
- Pone en riesgo la cuenta de Pepe si FA vincula el scraping

## Recomendación

**NO implementar scraping de perfiles FA.** El riesgo legal/reputacional no compensa, y técnicamente choca con las líneas rojas del proyecto.

**Alternativa recomendada: mantener el export RGPD como vía principal.** Ya funciona (verificado con el export real de Pepe), y es el camino que FA ha habilitado oficialmente. El flujo de usuario sería:

1. Usuario solicita su export RGPD en FA (tarda 24-72h según FA)
2. Recibe ZIP con `movie-ratings.html`
3. Sube el HTML a la webapp → matching → huecos

**Mejora UX posible:** la webapp puede incluir un tutorial visual paso a paso de cómo pedir el export RGPD en FA, para que el proceso sea lo más sencillo posible.

**Alternativa secundaria (extensión de navegador):** para usuarios avanzados dispuestos a instalarla, una extensión que lea la propia página de FA del usuario (con su sesión autenticada) y extraiga los votos. Esto NO es scraping externo — el usuario está leyendo su propia página. Pero requiere instalación y mantenimiento → solo como "nice to have", nunca como vía principal.

## Decisiones de implementación (abiertas, no bloquean borradores)

- ¿Descartamos el scraping de perfil público definitivamente? (recomendación: sí)
- ¿Incluimos la vía "extensión de navegador" como secundaria o la descartamos por carga de mantenimiento?

> Nota: la investigación está completa — scraping descartado. Solo queda decidir si la extensión de navegador merece el esfuerzo como vía alternativa.
