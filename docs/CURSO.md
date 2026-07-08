# FrameLanguageLM — curso a fondo

> Documento de referencia técnica. Asume que sabes qué es un transformer, un embedding, softmax y backprop. No asume que conozcas los trucos específicos de recomendación secuencial (SASRec, gBCE, evaluación full-ranking). Cada concepto se ancla con una película o serie real antes de formalizarlo.
>
> Basado en el estado del repo a 2026-07-07: fase 2 (baseline SASRec solo-ID) implementada y verificada en local; pendiente de entrenamiento completo en GPU. Fuentes: `SPEC.md`, `PLAN.md`, `STATUS.md`, `framelm/{data,model,loss,eval,train}.py`, `scripts/{smoke_test,debug_eval,vocab_analysis,build_vocab,build_catalog,fetch_tmdb}.py`.

---

## Índice

1. [La idea central](#1-la-idea-central)
2. [Arquitectura SASRec paso a paso](#2-arquitectura-sasrec-paso-a-paso)
3. [El problema del softmax con 100k ítems y la solución gBCE](#3-el-problema-del-softmax-con-100k-ítems-y-la-solución-gbce)
4. [El bug NaN: cuando el padding rompe la atención](#4-el-bug-nan-cuando-el-padding-rompe-la-atención)
5. [Evaluación honesta](#5-evaluación-honesta)
6. [Embeddings composicionales (fase 3, próxima)](#6-embeddings-composicionales-fase-3-próxima)
7. [Los datos](#7-los-datos)
8. [Inferencia y despliegue](#8-inferencia-y-despliegue)
9. [Glosario](#9-glosario)

---

## 1. La idea central

### 1.1 Un historial es una frase

Un modelo de lenguaje de texto aprende `P(palabra_t | palabra_1, ..., palabra_{t-1})`. Ve millones de frases y extrae la estructura estadística del lenguaje: después de "el gato se subió al", es mucho más probable "tejado" que "algoritmo".

FrameLanguageLM hace exactamente lo mismo, pero la "frase" es un historial de visionado y la "palabra" es una película o serie.

```
Frase de texto:      "el"  "gato"  "se"  "subió"  "al"  "tejado"
Frase de FrameLM:  "Mulholland Drive" → "Lost Highway" → "Inland Empire" → ?
```

El modelo entrenado sobre miles de historiales reales aprende que quien ve las dos primeras tiene una probabilidad alta de que la siguiente sea la tercera — no porque conozca el cine de Lynch como concepto, sino porque ha visto ese patrón de coocurrencia miles de veces en las secuencias de entrenamiento. Es la misma clase de conocimiento estadístico que un LM de texto tiene sobre "tejado" tras "se subió al", trasladada de palabras a títulos.

El paralelismo es exacto, no una metáfora:

| LM de texto | FrameLanguageLM |
|---|---|
| Vocabulario = palabras/subpalabras (~50k tokens) | Vocabulario = películas/series (100k títulos) |
| Frase = secuencia de tokens | Historial = secuencia de títulos vistos, en orden temporal |
| Tarea = predecir el siguiente token | Tarea = predecir la siguiente película bien valorada |
| Token especial `<pad>` | Token especial `<pad>` (mismo rol exacto: relleno hasta longitud fija) |
| Embedding de token + posición | Embedding de ítem + posición (§2) |
| Softmax final sobre el vocabulario | Softmax (aproximado, §3) sobre el catálogo |

La diferencia de fondo no está en la arquitectura — está en la escala y en la "gramática". El lenguaje tiene una estructura composicional profunda (sintaxis recursiva, ambigüedad, dependencias a larga distancia entre cláusulas). El "lenguaje del gusto cinematográfico" es mucho más plano: patrones de coocurrencia y proximidad temática, sin una gramática generativa comparable. Esto tiene consecuencias directas en el tamaño del modelo que hace falta (§2.4).

### 1.2 El usuario ES su secuencia

Aquí está la decisión de diseño que más se aleja de la intuición de "un sistema de recomendación típico". Los sistemas clásicos de filtrado colaborativo (matrix factorization, por ejemplo) aprenden un **vector de usuario** fijo: `u_Pepe ∈ R^d`, entrenado junto a los vectores de ítem, y la predicción es `score(item) = u_Pepe · v_item`.

SASRec no tiene ese vector. No existe `nn.Embedding` para usuarios en `framelm/model.py` — búscalo, no está. El modelo solo tiene:

```python
self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
self.pos_emb  = nn.Embedding(max_len, d)
```

La "identidad" de un usuario nuevo, con un historial que el modelo nunca ha visto exactamente, no vive en ningún parámetro entrenado. Vive en la secuencia que le pasas por el forward pass. El usuario **es** literalmente la secuencia de títulos que introduces — nada más, nada menos.

Esto tiene una consecuencia práctica enorme, y es la base de todo el caso de uso 1 del `SPEC.md` (recomendación in-context, D5): no hace falta re-entrenar nada para dar recomendaciones a un usuario nuevo. Le pasas su lista de Letterboxd exportada, un forward pass, y el modelo ya "sabe" quién es — de la misma forma en que un LLM sin fine-tuning puede continuar cualquier texto nuevo que le des como prompt, sin haber visto jamás esa frase exacta antes. Es in-context learning, aplicado a gustos de cine en vez de a lenguaje.

```
┌─────────────────────────────────────────────────────────┐
│  Filtrado colaborativo clásico     │  SASRec (FrameLM)   │
│  ─────────────────────────────     │  ─────────────────  │
│  u_Pepe = vector aprendido fijo    │  "Pepe" = [El Norte, │
│  score = u_Pepe · v_item           │   Roma, Amores       │
│  usuario nuevo → sin vector        │   Perros, ...]       │
│  → problema de cold-start          │  score = f(secuencia)│
│                                     │  usuario nuevo → OK  │
└─────────────────────────────────────────────────────────┘
```

El precio de esta elección: el modelo solo "conoce" al usuario dentro de la ventana de contexto (`max_len=200` en este proyecto). Si el historial es más largo, lo que quedó fuera de la ventana no influye en la predicción — igual que un LLM no recuerda lo que dijiste fuera de su contexto.

---

## 2. Arquitectura SASRec paso a paso

Código de referencia: `framelm/model.py`, 71 líneas. La brevedad es la primera lección: esto es deliberadamente ~500 líneas en todo el repo (decisión D10 del `SPEC.md`), no un framework.

### 2.1 Embedding de ítem + posicional

```python
self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)   # d=256
self.pos_emb  = nn.Embedding(max_len, d)                       # max_len=200

h = self.item_emb(seq) + self.pos_emb(pos)      # (B, L, d)
```

Cada posición de la secuencia recibe la suma de dos vectores de 256 dimensiones: "qué película es" (`item_emb`) y "en qué posición de la secuencia está" (`pos_emb`). Exactamente el mismo mecanismo que GPT: sin el embedding posicional, la atención (§2.2) es invariante a permutaciones — al mecanismo de atención en sí le da igual el orden, así que hay que inyectarlo explícitamente.

`padding_idx=0` le dice a PyTorch dos cosas: que el índice 0 está reservado para `<pad>`, y que su gradiente se congela en cero — el vector de padding nunca se actualiza, se queda anclado a donde se inicializó (y el código lo inicializa explícitamente a cero: `self.item_emb.weight[0].zero_()`).

### 2.2 Atención causal multi-head, con la matemática

Cada `Block` (hay `n_layers=2` de ellos) hace:

```python
x = self.ln1(h)
a, _ = self.attn(x, x, x, attn_mask=causal, key_padding_mask=pad_mask)
h = h + self.drop(a)
h = h + self.drop(self.ffn(self.ln2(h)))
```

Desglosemos `self.attn(x, x, x, ...)`, que es una `nn.MultiheadAttention`. Para cada cabeza `i` (aquí `n_heads=2`, cada una con dimensión `d_k = d/n_heads = 128`):

```
Q_i = x · W_Q_i          K_i = x · W_K_i          V_i = x · W_V_i

Atención_i = softmax( (Q_i · K_i^T) / √d_k  +  mask ) · V_i
```

`x` tiene forma `(B, L, d)` — para cada posición de la secuencia, un vector de 256. `Q_i, K_i, V_i` son proyecciones lineales de ese mismo `x` (por eso es *self*-attention: la secuencia se consulta a sí misma). `Q_i · K_i^T` da una matriz `(L, L)`: para cada par de posiciones `(t, s)`, cuánto "encaja" lo que busca la posición `t` (su query) con lo que ofrece la posición `s` (su key). Se divide por `√d_k` para que la varianza de esos productos punto no crezca con la dimensión y el softmax no se sature (el truco estándar de "scaled" dot-product attention).

**La máscara causal.** Sobre esa matriz `(L, L)` se suma una máscara antes del softmax:

```python
causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
```

`torch.triu(..., diagonal=1)` marca `True` todo lo estrictamente por encima de la diagonal — las posiciones futuras. PyTorch convierte esas posiciones marcadas a `-inf` antes del softmax, así que `exp(-inf) = 0`: la posición `t` no puede atender a ninguna posición `s > t`.

```
        Mulholland  Lost Highway  Inland Empire  ?
Mulholland    ✓           ✗             ✗        ✗
Lost Highway  ✓           ✓             ✗        ✗
Inland Emp.   ✓           ✓             ✓        ✗
```

Sin esto, el modelo "haría trampa": al predecir qué viene después de *Mulholland Drive*, podría mirar directamente *Inland Empire* (que está más adelante en la secuencia) en vez de aprender a inferirlo del pasado. Es idéntico al *causal masking* de GPT — aquí la analogía con LMs de texto no es una simplificación pedagógica, es literalmente el mismo mecanismo.

Las salidas de las `n_heads` cabezas se concatenan y se proyectan de vuelta a dimensión `d` (esto lo hace `nn.MultiheadAttention` internamente). Cada cabeza puede especializarse en un tipo de relación distinto — una podría capturar "mismo director", otra "misma década" — sin que nadie se lo imponga explícitamente; emerge del entrenamiento.

### 2.3 Pre-LayerNorm y la conexión residual

Fíjate en el orden exacto:

```python
x = self.ln1(h)                     # normalizar ANTES de atender
a, _ = self.attn(x, x, x, ...)
h = h + self.drop(a)                # residual: sumar sobre h, no sobre x
```

Esto es **pre-LN** (LayerNorm antes del sub-bloque, no después). La alternativa clásica del *Attention Is All You Need* original era post-LN (`LayerNorm(h + attn(h))`). Pre-LN es lo que casi todos los transformers modernos usan porque estabiliza el entrenamiento sin necesidad de *warmup* de learning rate cuidadosamente calibrado — el camino residual (`h + ...`) queda "limpio", sin pasar por una normalización, lo que evita que el gradiente se atenúe al retropropagar por muchas capas. Con solo `n_layers=2` esto importa menos que en un LLM de 96 capas, pero es la opción robusta por defecto y no cuesta nada.

La FFN (feed-forward network) tras la segunda normalización:

```python
self.ffn = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
h = h + self.drop(self.ffn(self.ln2(h)))
```

Es donde el modelo hace cómputo *por posición* (a diferencia de la atención, que mezcla información *entre* posiciones). Nota que aquí no hay expansión a `4d` como en un transformer de texto estándar (`d → 4d → d`) — se queda en `d → d → d`. Con un vocabulario y una tarea mucho más simples que el lenguaje, no hace falta esa capacidad extra.

### 2.4 Weight tying: por qué la matriz de scoring es la misma que la de entrada

```python
def score_all(self, h_last: torch.Tensor) -> torch.Tensor:
    return h_last @ self.item_emb.weight.T
```

No hay una matriz de salida separada. Para puntuar cada película del catálogo, el modelo hace el producto punto entre el estado oculto final (`h_last`, de dimensión `d`) y **la misma tabla de embeddings de entrada**, transpuesta: `(d,) @ (d, n_items+1) → (n_items+1,)`.

Esto es *weight tying*, la misma técnica que usan GPT-2 y la mayoría de LMs de texto entre el embedding de entrada y la proyección de salida. La intuición: "qué tan probable es que la película X sea la siguiente" y "qué representa la película X" deberían vivir en el mismo espacio vectorial — si el modelo aprende que `E(Lost Highway)` está cerca de `E(Mulholland Drive)`, esa cercanía sirve tanto para *entender* una secuencia que contiene una de las dos como para *puntuar alto* la otra al predecir. Separar ambas matrices duplicaría 25.6M parámetros (100k × 256) sin ganancia clara, y en este proyecto en concreto es además necesario para el weight tying del §6: cuando la matriz de embeddings deje de ser una tabla libre y pase a ser `E_ID + proj(features)`, seguir puntuando con la misma matriz es lo que hace que la búsqueda paramétrica y el cold-start (§6) funcionen con el mismo mecanismo de scoring, sin una cabeza de salida aparte que entrenar.

### 2.5 Por qué 2 capas y d=256 bastan aquí (y no en un LLM)

Tres razones, todas cuantificables con los números reales del proyecto:

1. **Vocabulario y "gramática" mucho más simples.** GPT-3 modela sintaxis, semántica composicional, razonamiento encadenado sobre 50k+ subpalabras con una estructura recursiva profunda. FrameLanguageLM modela "qué títulos coocurren cerca en el tiempo" sobre como mucho 100k ítems (54k con señal real de entrenamiento, ver §7) — un problema con muchísima menos profundidad estructural. El paper original de SASRec (Kang & McAuley, 2018) y su reanálisis en gSASRec (Petrov & Macdonald, RecSys'23) encuentran empíricamente que apilar más de 2-3 bloques no mejora las métricas en recomendación secuencial estándar, y a veces las empeora por sobreajuste.

2. **Secuencias cortas.** `max_len=200` frente a contextos de miles o decenas de miles de tokens en un LLM moderno. Menos posiciones que relacionar, menos necesidad de que capas profundas compongan dependencias de largo alcance.

3. **Volumen de datos de entrenamiento.** 31.6M interacciones (`STATUS.md`) es un corpus minúsculo comparado con los billones de tokens de un LLM. Un modelo con más capacidad que la que este volumen de datos puede sostener simplemente memoriza ruido — sobreajusta. `d=256` con 2 capas ya da un modelo de unos pocos millones de parámetros (el bloque grande son los propios embeddings: `(100k+1) × 256 ≈ 25.6M` solo en `item_emb`), acorde al rango de 5-30M que fija el `SPEC.md` como objetivo explícito de la Visión (§1).

---

## 3. El problema del softmax con 100k ítems y la solución gBCE

### 3.1 Por qué el softmax completo es un problema

Para entrenar por *next-token prediction* de forma "correcta", cada paso necesitaría un softmax completo sobre el catálogo:

```
P(item = i | contexto) = exp(score_i) / Σ_{j=1}^{100k} exp(score_j)
```

El numerador es barato (un producto punto). El denominador exige calcular `score_j` para los 100k ítems, en cada una de las `L=200` posiciones de cada secuencia del batch, en cada paso de entrenamiento. No es intratable a esta escala (a diferencia de un LLM con 100k+ tokens de vocabulario reales, aquí sí cabría en memoria), pero sí es caro, y el `SPEC.md` (D3) lo deja como baseline a probar "si sobra VRAM" — no como opción por defecto.

La alternativa estándar y mucho más barata: en vez de comparar contra *todo* el catálogo, compara el ítem positivo contra un puñado de negativos muestreados al azar, y entrena con **BCE** (binary cross-entropy): "¿es más probable el positivo que este negativo aleatorio?", repetido `k` veces por posición.

### 3.2 Por qué el BCE con negativos muestreados sobreestima

Aquí está el problema, y es sutil. BCE con negative sampling optimiza una tarea *más fácil* que el softmax completo: distinguir el positivo de `k` negativos aleatorios, no de los 100k−1 competidores reales. Si `k=256` y `n_items≈100k`, el modelo casi nunca ve como negativo a un competidor genuinamente difícil (una película muy parecida a la correcta) — la mayoría de negativos aleatorios son triviales de descartar.

El resultado: el modelo aprende a asignar probabilidades (vía sigmoide) sistemáticamente más altas al positivo de lo que le correspondería frente a la competencia real. Es *overconfidence* — no porque el modelo esté mal entrenado, sino porque la tarea de entrenamiento (vencer a `k` negativos fáciles) es estructuralmente más fácil que la tarea real (vencer a todo el catálogo), y el sesgo se cuela en las probabilidades.

Ejemplo concreto: el modelo ve `Mulholland Drive → Lost Highway` como positivo, y como negativos, 256 títulos aleatorios del catálogo — pongamos que le toca *Fast & Furious 9* entre ellos. Distinguir *Lost Highway* de *Fast & Furious 9* es trivial. El modelo nunca tiene que aprender a distinguirlo de un negativo *duro* de verdad, como *Inland Empire* o *Eraserhead* — títulos con los que sí competiría en un ranking real. Sin corrección, el score que aprende a darle a *Lost Highway* está inflado respecto al que tendría en una comparación justa contra los 100k.

### 3.3 gBCE: la corrección de Petrov & Macdonald (RecSys'23)

La solución del paper de gSASRec no es cambiar el muestreo — es corregir matemáticamente el sesgo que ese muestreo introduce, elevando el término positivo de la BCE a una potencia `β` calibrada:

```python
alpha = k / (n_items - 1)
beta  = (1 - t) + t * alpha

pos_term = beta * F.softplus(-pos_scores)                # (B, L)
neg_term = F.softplus(neg_scores).sum(-1)                 # (B, L)
per_pos  = (pos_term + neg_term) / (1 + k)
```

Formalmente: en vez de optimizar `log σ(s_pos)` (BCE estándar), gBCE optimiza `log σ(s_pos)^β = β · log σ(s_pos)`. Como la pérdida es la log-verosimilitud negativa, y `-log σ(s) = softplus(-s)` (identidad estándar de la sigmoide), el término positivo queda `β · softplus(-s_pos)` — exactamente la línea del código. Para el término negativo, `-log(1 - σ(s_neg)) = -log σ(-s_neg) = softplus(s_neg)`, sumado sobre los `k` negativos.

**La intuición de `β`.** `alpha = k / (n_items - 1)` es, aproximadamente, la fracción del catálogo que efectivamente se muestrea como competencia en cada paso (con `k=256` y `n_items≈54k-100k`, un valor muy pequeño — del orden de 0.25%-0.5%). `β` interpola entre dos extremos usando el parámetro `t`:

- `t = 0`: `β = 1`, BCE normal, sin corrección — el sesgo de overconfidence queda intacto.
- `t = 1`: `β = alpha`, la corrección máxima — el término positivo se atenúa en proporción a lo pequeña que es la muestra de negativos frente al catálogo real. Esto es lo que el paper demuestra que hace que la distribución de probabilidades aprendida converja, en el límite, a la que daría el softmax completo — es decir, `t=1` recupera la calibración del softmax completo sin pagar su coste computacional.

El proyecto usa `t=0.75` (default en `framelm/train.py`), un punto intermedio: corrige la mayor parte del sesgo sin llevar `β` hasta el extremo teórico, que en la práctica puede ser numéricamente más inestable con `k` pequeño.

**Por qué 256 negativos uniformes.** El propio mecanismo de `β` es lo que permite usar un `k` pequeño y barato (256, no miles) y aun así obtener probabilidades bien calibradas — la corrección compensa matemáticamente la falta de negativos, en vez de intentar acercarse al softmax completo a fuerza de muestrear más. Es la razón de ser de todo el método: gBCE existe precisamente para que `k=256` sea suficiente. El código además comparte los mismos `k` negativos entre todas las posiciones de una fila del batch (`sample_negatives(batch_size, k, ...)`, no por posición) — ahorra memoria, y la colisión entre un negativo muestreado y el positivo real de esa posición tiene probabilidad `~k/n_items`, despreciable con un catálogo de decenas de miles.

---

## 4. El bug NaN: cuando el padding rompe la atención

Esta sección documenta un bug real que apareció durante la implementación del baseline (commit `8656191`, 2026-07-07) y el razonamiento para diagnosticarlo y arreglarlo. Vale la pena entenderlo a fondo porque el mecanismo es genérico — reaparece en cualquier transformer con padding y atención causal, no solo aquí.

### 4.1 El mecanismo: por qué una fila puede quedar 100% enmascarada

Las secuencias se rellenan con **padding a la izquierda** (`left_pad` en `framelm/data.py`): si un usuario solo tiene 4 ítems vistos y `max_len=200`, el vector de entrada tiene 196 ceros seguidos de los 4 ítems reales al final.

```
posición:    0    1    2   ...  195  196  197  198  199
secuencia:  pad  pad  pad ...  pad   5    9    3    7
```

Ahora combina esto con la máscara causal (§2.2): en la posición 0 (la más a la izquierda), la máscara causal solo permite atender a la propia posición 0 (ninguna posición futura). Pero la posición 0 *es* padding — el `key_padding_mask` la marca como "ignorar". El resultado: para la fila de atención correspondiente a la posición 0, **no queda ninguna key válida a la que atender**. Causal permite solo `{0}`; padding excluye `{0}`. La intersección es el conjunto vacío.

Antes del softmax, esa fila tiene todos sus logits en `-∞`. `softmax` de una fila de puros `-∞` es matemáticamente `0/0`: **NaN**, no cero. Esto no es un caso raro — con `max_len=200` y usuarios de MovieLens que a menudo tienen historiales cortos, es la situación *típica*, no la excepción.

### 4.2 Por qué "simplemente multiplicar por la máscara" no basta

La reacción intuitiva es: "vale, esas posiciones de padding no me importan, las pongo a cero después y ya está":

```python
h = h * keep    # keep = (~pad_mask), 0 en las posiciones de padding
```

Esto **no funciona** si `h` ya contiene NaN en esas posiciones. En aritmética IEEE754, `0 * NaN = NaN`, no `0`. El comentario del código lo dice explícitamente:

```python
# posiciones pad: filas de atencion 100% enmascaradas -> softmax NaN,
# y 0*NaN contamina el resto via values. Se anulan tras cada bloque.
```

La solución real es sanear explícitamente antes de enmascarar:

```python
h = torch.nan_to_num(h) * keep    # NaN -> 0 primero, LUEGO multiplicar
```

### 4.3 Por qué "vía values" — el contagio no es donde parece

Aquí está la parte no obvia. Dentro de un único forward pass, el NaN de la posición 0 **no debería** filtrarse hacia las posiciones válidas (196-199): el `key_padding_mask` impide que cualquier posición válida use la posición 0 como *key* — nadie atiende a una key marcada como padding, sea cual sea su valor. Y las capas de `LayerNorm` y la FFN operan posición a posición (normalizan y transforman el último eje, `d`), así que tampoco mezclan información entre posiciones de la secuencia. En principio, el NaN debería quedar aislado en las posiciones de padding, sin tocar nunca el resultado que de verdad importa (`h_last`, la última posición, que por construcción de las secuencias siempre es un ítem real).

El contagio real ocurre en el **backward pass**, a través de los parámetros *compartidos*. Las proyecciones `W_Q, W_K, W_V` de la atención (y los pesos de la FFN) son las mismas para todas las posiciones y todo el batch — no hay una copia por posición. El gradiente de la pérdida respecto a, por ejemplo, `W_Q` es una suma sobre *todas* las posiciones de *todo* el batch:

```
∂L/∂W_Q = Σ_{b,l}  x[b,l] ⊗ ∂L/∂Q[b,l]
```

Si la posición `l` es una de esas filas 100%-enmascaradas, su contribución local a `∂L/∂Q[b,l]` es NaN (viene del softmax NaN de esa fila, propagado hacia atrás por la regla de la cadena). Aunque esa posición nunca influyó en la pérdida final por el camino "hacia adelante" (está enmascarada en la pérdida también — `gbce_loss` filtra con `targets != 0`), el grafo de autograd sigue calculando un gradiente local para ella, y ese NaN se **suma** dentro de la misma reducción que produce el gradiente de `W_Q` para *todas* las posiciones. Un solo `NaN` en esa suma contamina el gradiente entero. Tras `opt.step()`, `W_Q` completo pasa a ser NaN — y como esos pesos son compartidos, el modelo entero queda envenenado para *todas* las posiciones, de todos los usuarios, para siempre (o hasta reiniciar desde un checkpoint limpio).

Esto es lo que dice "contamina el resto **vía values**" del comentario: no es un contagio espacial dentro de la secuencia (eso está bien aislado por el masking), es un contagio a través de los pesos compartidos que todas las posiciones usan como *value*/proyección — de ahí que haya que limpiar el NaN **tras cada bloque**, antes de que el grafo de cómputo acumule más operaciones sobre él.

### 4.4 Cómo esto produjo un NDCG fantasma

El síntoma que expuso el bug no fue un crash — fue una métrica de validación sospechosamente alta. La razón, siguiendo `framelm/eval.py`:

```python
scores = model.score_all(h_last)          # (B, n+1)
...
tgt_scores = scores.gather(1, tgt.unsqueeze(1))
ranks = (scores > tgt_scores).sum(1)      # 0-based
```

Si para un usuario con historial extremadamente corto `h_last` mismo resultaba NaN (porque incluso la última posición, en algún caso límite, heredaba NaN antes de la limpieza), entonces `scores` para ese usuario es NaN en **todas** las columnas. La comparación `scores > tgt_scores` con ambos lados NaN evalúa a `False` en cada entrada — en IEEE754, cualquier comparación con NaN es `False`, incluida `NaN > NaN`. El resultado: `ranks = 0` para ese usuario. Un rank de 0 es, según la métrica, *la predicción perfecta*: el ítem objetivo "gana" a los 100k competidores. Ese usuario contribuye el máximo posible a NDCG@10 (`1/log2(0+2) = 1.0`) sin que el modelo haya aprendido nada real — es un artefacto de que `NaN > x` es siempre falso, no una señal de que el ranking fuera correcto.

Con suficientes usuarios de historial corto afectados, la métrica agregada se infla notablemente por encima de lo que el modelo realmente sabe hacer — de ahí el "NDCG 0.80 fantasma": un número que parece un resultado excelente y es, en realidad, el efecto colateral de un bug de punto flotante.

### 4.5 La lección: los smoke tests sintéticos no bastan

`scripts/smoke_test.py` sí comprueba explícitamente la ausencia de NaN (`assert not torch.isnan(h).any()`) sobre secuencias sintéticas cortas — y hoy pasa, porque el fix ya está en el modelo. Pero esa comprobación mecánica (¿el forward produce NaN sobre un batch de juguete?) es una condición necesaria, no suficiente, para confiar en una métrica.

Lo que expuso el problema no fue el smoke test — fue `scripts/debug_eval.py`, un script separado escrito específicamente para *desconfiar* de un número bueno: comprobar si hay NaN en los scores reales, mirar la distribución de ranks (¿cuántos son exactamente 0? ¿es sospechoso?), y contrastar contra la estructura temporal real de los datos (§5.4). Un test sintético con un patrón cíclico limpio (`i → i+1`, como el que usa `test_loss_decreases`) no reproduce ni la distribución real de longitudes de historial de MovieLens (muchos usuarios con muy pocos ratings, forzando padding extremo) ni sus artefactos temporales. La regla general: un test que verifica mecánica (no revienta, el loss baja, las formas cuadran) no verifica que el *número final* signifique lo que crees que significa. Eso solo se descubre inspeccionando predicciones reales sobre datos reales, con la sospecha activa de que un resultado "demasiado bueno" probablemente lo sea.

---

## 5. Evaluación honesta

### 5.1 Leave-one-out temporal

Para cada usuario, la secuencia ordenada por timestamp se corta en tres:

```python
# framelm/data.py — eval_batches()
if mode == "valid":
    ctx, tgt = s[:-2], s[-2]     # penúltimo item = target de validación
else:  # test
    ctx, tgt = s[:-1], s[-1]     # último item = target de test
```

```
secuencia completa:  [t1, t2, t3, ..., t_{n-2}, t_{n-1}, t_n]
                      └──────── train ────────┘   valid    test
```

`TrainDataset` usa `s[:-2]` — el entrenamiento nunca ve ni el target de validación ni el de test, ni siquiera como parte del contexto de otra posición. Esto es crítico: sin este corte, información del futuro (relativo al punto de evaluación) se filtraría al entrenamiento y las métricas estarían infladas por una fuga temporal, no por capacidad real del modelo.

### 5.2 Full-ranking, sin sampling de candidatos — y por qué importa

La forma "barata" de evaluar recomendadores, históricamente muy extendida, es: coger el ítem objetivo, mezclarlo con (digamos) 100 negativos aleatorios, rankear esos 101, y medir NDCG/Recall sobre ese ranking reducido. `framelm/eval.py` explícitamente no hace esto:

```python
scores = model.score_all(h_last)      # (B, n+1) — TODO el catálogo
scores[:, 0] = float("-inf")
scores[seen] = float("-inf")          # excluir lo ya visto
ranks = (scores > tgt_scores).sum(1)  # rank real, contra TODO n_items
```

Cada usuario se rankea contra el catálogo completo (menos lo que ya ha visto y el padding). Esto es más caro — para cada usuario del batch de validación hay que puntuar el vocabulario entero — pero es la métrica correcta.

**Por qué el sampled evaluation está desacreditado.** Krichene & Rendle, *"On Sampled Metrics for Item Recommendation"* (KDD 2020), demostraron formalmente que las métricas calculadas sobre un subconjunto muestreado de negativos no son un estimador consistente de la métrica real (full-ranking) — y, peor, que el error introducido no es uniforme entre modelos: un modelo puede parecer mejor que otro bajo evaluación muestreada y peor bajo evaluación completa, invirtiendo el ranking de qué modelo es realmente superior. Es decir, no es solo "menos preciso" — puede llevar a decisiones erróneas sobre qué arquitectura o hiperparámetro elegir. El `SPEC.md` (§4.5) lo cita explícitamente como referencia a evitar, y es la razón de que `evaluate()` pague el coste computacional del ranking completo.

### 5.3 Qué miden NDCG@10 y Recall@k exactamente

Con `rank` definido como el número de ítems que superan en score al target verdadero (0-indexado: `rank=0` significa que el target es el mejor puntuado de todo el catálogo):

```python
recall10 += (ranks < 10).sum()             # ¿cayó el target en el top-10?
ndcg10   += (1/log2(ranks+2))[ranks<10].sum()
```

- **Recall@k**: fracción de usuarios cuyo ítem objetivo real cae dentro del top-k del ranking completo. Es binario por usuario (0 o 1), promediado. Recall@10=0.15 significa: "en el 15% de los casos, la película que el usuario vio a continuación estaba entre las 10 mejores predicciones del modelo, de entre ~54k-100k candidatos".
- **NDCG@10**: como Recall, pero pondera *dónde* dentro del top-10 cae el acierto. `1/log2(rank+2)` da el máximo peso (1.0) cuando `rank=0` (el target es la predicción número 1), y decae logarítmicamente cuanto más abajo aparece dentro del top-10; fuera del top-10 no suma nada. Aquí no hace falta la normalización habitual (la "N" de NDCG, dividir por el DCG ideal) porque solo hay **un** ítem relevante por usuario en este protocolo — el DCG ideal es siempre exactamente 1.0 (acertar en la primera posición), así que DCG y NDCG coinciden.

### 5.4 El 87%: sesiones de volcado en MovieLens

`scripts/debug_eval.py` incluye un test de hipótesis específico, motivado por la sospecha de fuga temporal tras el episodio del NDCG fantasma (§4.4): ¿el target de validación de un usuario ocurre genuinamente *después*, en el tiempo, del último ítem de su contexto — o comparte, de hecho, casi el mismo instante?

```sql
avg(CASE WHEN t1.timestamp - t2.timestamp < 300 THEN 1.0 ELSE 0.0 END) AS within_5min
```

El hallazgo: aproximadamente el **87%** de los targets están a menos de 5 minutos del ítem inmediatamente anterior en la secuencia del mismo usuario. MovieLens registra el *timestamp del rating*, no el del visionado — y es sabido en la literatura de recsys que muchos usuarios de MovieLens puntúan en sesiones de "volcado": se sientan una vez y valoran de golpe decenas de películas que vieron en momentos muy distintos de su vida, todas con timestamps casi idénticos porque el acto de puntuar (no de ver) fue simultáneo.

**Qué implica esto.** No invalida el entrenamiento — la coocurrencia de qué títulos un mismo usuario puntúa juntos sigue siendo señal real y útil de gusto compartido (es, de hecho, la señal que exploran los propios sistemas de filtrado colaborativo desde su origen). Pero sí matiza la narrativa de "secuencia = orden real de visionado" del §1: gran parte del "siguiente ítem" en el split leave-one-out no es literalmente "lo próximo que esta persona vio", es "lo próximo que puntuó en la misma sesión de volcado". El modelo puede estar aprendiendo, en una fracción sustancial de casos, patrones de *coocurrencia dentro de una sesión de rating* más que dependencia *secuencial temporal genuina* a través del tiempo — algo a tener en cuenta al interpretar qué tan lejos llega la analogía con un LM de texto (donde el orden de las palabras sí es el orden real de producción del lenguaje, no un artefacto de cuándo se registró cada una).

---

## 6. Embeddings composicionales (fase 3, próxima)

Esta fase no está implementada aún (`STATUS.md`: "No pasar a fase 3 hasta tener [el] baseline [de fase 2]"). Lo que sigue es el diseño fijado en `SPEC.md` (D2, §4.2) y su justificación.

### 6.1 La fórmula y por qué se suma, no se reemplaza

```
E(item) = E_ID  +  W · concat(E_director, mean(E_cast), E_géneros, E_país, E_idioma, E_década, E_budget_bucket)
```

`E_ID` es exactamente lo que ya existe hoy: la fila libre de `item_emb`, entrenada solo a partir de coocurrencias reales (§2.1). La parte composicional es una proyección lineal (`W`) sobre la concatenación de embeddings de metadata, cada uno de ellos también entrenado, pero compartido entre *todos* los ítems que tienen ese mismo director, ese mismo género, esa misma década.

**Por qué sumar y no sustituir.** Si el embedding final de un ítem fuera *solo* la parte composicional, dos películas con la misma ficha técnica (mismo director, género, década, país) tenderían a colapsar al mismo punto del espacio, perdiendo toda la información fina que sí capturó `E_ID` a partir del patrón real de quién-vio-qué. Ese riesgo es lo que el SPEC llama "invasión de información": dejar que la metadata *domine* sobre la señal colaborativa aprendida. Sumar en vez de sustituir dice: el ID sigue siendo la fuente principal de verdad cuando hay señal colaborativa real que aprender de ella, y la metadata añade una corrección — información adicional, no un reemplazo. Para un ítem "caliente" (con mucha señal de entrenamiento), `E_ID` domina el vector final y la parte composicional aporta un empujón fino. Para un ítem "frío" (sin ninguna señal), es la única parte que aporta algo con sentido (§6.2).

### 6.2 Cold-start: por qué esto resuelve a los ~46k títulos sin señal

Un ítem que nunca aparece como positivo ni como negativo en ningún batch de entrenamiento nunca recibe gradiente en su fila de `E_ID` — se queda exactamente en su inicialización aleatoria (`std=0.02`), indistinguible del ruido. Con el modelo actual (solo-ID), eso significa que los ~46k títulos del catálogo sin señal de MovieLens (§7.3) — entre ellos, **todas** las series, por decisión D7 — no tienen ningún embedding útil. Cualquier ranking o búsqueda de vecinos sobre ellos sería arbitrario.

La parte composicional rompe esa dependencia. `W` y los embeddings de director/cast/género/país/etc. **sí** reciben gradiente cada vez que se entrena con *cualquier* ítem que comparta esos atributos — y como la inmensa mayoría de directores, géneros y décadas del catálogo sí aparecen representados por al menos algún título con señal real, esos componentes acaban bien entrenados incluso para combinaciones de metadata que nunca vieron un ítem concreto con entrenamiento directo. El resultado: para un ítem frío, `E(item) = ruido_aleatorio + proj(features_bien_entrenadas) ≈ proj(features)` — un embedding no degenerado, situado sensatamente cerca de otros títulos con ficha técnica parecida, aunque el ítem en sí jamás haya aparecido en una secuencia de entrenamiento. Es exactamente el mecanismo que hace viable recomendar series (sin señal colaborativa por D7) o títulos recientes fuera del rango de MovieLens (riesgo R2 del SPEC) con una calidad razonable.

Esto mismo es lo que sostiene la **búsqueda paramétrica** (caso de uso 2 del SPEC): consultas como "como *Mulholland Drive* pero francesa" solo tienen sentido si *todo* el catálogo — no solo el 54% con señal de entrenamiento — vive en un espacio de embeddings coherente donde la distancia significa algo.

### 6.3 Rating-buckets en el input

```
E(posición) = E(item) + E(rating_bucket) + E(posición)
```

Cada posición de la secuencia de entrada añade un tercer embedding: en qué bucket cayó la valoración explícita que el usuario dio a ese ítem (`≤2`, `2.5-3`, `3.5-4`, `4.5-5`, `sin-rating`). Esto le da al modelo una señal que el solo-ID de fase 2 descarta por completo: la diferencia entre "lo vi y me encantó" y "lo vi y me dejó indiferente" — dos eventos que hoy, sin rating-bucket, son indistinguibles en la secuencia de entrada.

Se combina con un segundo mecanismo, en el *target*, no en el input: el objetivo de predicción en cada posición deja de ser "el siguiente ítem visto" y pasa a ser "el siguiente ítem visto **y bien valorado** (rating ≥3.5)" — un filtro suave. La razón es de producto, no solo técnica: el objetivo final (SPEC §1, caso de uso 1) es recomendar huecos que probablemente *gusten*, no simplemente huecos que probablemente se *vean*. Sin este filtro, el modelo optimizaría fielmente por predecir el siguiente consumo, incluida la película que el usuario vio y odió — una señal que no queremos amplificar.

---

## 7. Los datos

### 7.1 MovieLens: 98.8% de concentración

MovieLens 32M es un dataset estándar de investigación en recsys: 32 millones de ratings explícitos (escala 0.5-5) de 200k usuarios sobre películas, con timestamp por rating (Harper & Konstan, 2015). `links.csv` mapea cada `movieId` interno de MovieLens a su `imdbId`/`tmdbId`, que es el puente que usa `build_catalog.py` para unir MovieLens con el catálogo IMDb/TMDB.

`scripts/vocab_analysis.py` mide, para distintos cortes de vocabulario (por `numVotes` de IMDb), qué fracción de los 32M ratings de MovieLens cae dentro de ese corte. El resultado que fijó la decisión del `PLAN.md` (§ Decisiones, punto 1): el corte de **100k títulos** (los más votados en IMDb, tipos `movie`/`tvSeries`/`tvMiniSeries`) retiene el **98.8%** de todos los ratings de MovieLens.

Esto no es casualidad — es la ley de potencias habitual en consumo cultural: la inmensa mayoría de lo que la gente valora está concentrado en un núcleo relativamente pequeño de títulos populares, con una cola larguísima de títulos con pocos o ningún rating. Cortar en 100k no es una limitación forzada por recursos; es, casi literalmente, capturar prácticamente todo lo que hay que capturar de MovieLens y dejar fuera solo la cola que apenas aporta señal.

### 7.2 Qué aporta cada fuente

| Fuente | Aporta | Por qué hace falta |
|---|---|---|
| **IMDb** (datasets no comerciales, TSV) | `titleType`, título, año, géneros, `numVotes` (para el corte de vocabulario), director (`title.crew`), reparto top-6 (`title.principals`, ordenado por `ordering`) | Es la columna vertebral de identidad y taxonomía — pero el volcado plano de IMDb **no incluye** país, idioma original ni presupuesto |
| **TMDB** (API en vivo) | País de producción, idioma original, presupuesto (bucketizado), keywords, poster, popularidad | Estos campos simplemente no existen en el dump no-comercial de IMDb; hace falta pedirlos activamente por API (`scripts/fetch_tmdb.py`, ~100k requests, reanudable) |

### 7.3 El vocabulario efectivo: 54k vs 100k

`data/vocab_map.json` fija 100k índices para el catálogo elegido por popularidad IMDb. Pero `STATUS.md` documenta una cifra distinta: solo **54.053** de esos 100k ítems tienen alguna interacción real en `sequences.parquet` — es decir, aparecen al menos una vez en el corpus de entrenamiento de MovieLens tras el join.

La diferencia (~46k ítems) no es un error — es la consecuencia directa de dos hechos combinados:

1. MovieLens es un dataset **solo de películas**. No contiene ni una sola valoración de una serie de televisión.
2. El catálogo de 100k incluye, por diseño, ~21k series (decisión D6/D7 del SPEC).

Todas las series, más una parte de películas de nicho o muy recientes (posteriores al corte de MovieLens en octubre de 2023, riesgo R2) que IMDb sí indexa pero MovieLens nunca llegó a registrar, caen fuera de esos 54k. Con el modelo actual (solo-ID, fase 2), esos ~46k títulos son, en la práctica, ruido sin entrenar — el hueco exacto que la fase 3 (§6) está diseñada para cerrar mediante cold-start composicional. La cifra "54k vs 100k" es, en otras palabras, la medida concreta de cuánto trabajo le queda a la fase 3 antes de que el catálogo completo tenga embeddings con sentido.

---

## 8. Inferencia y despliegue

### 8.1 Por qué brute-force gana a un índice ANN a esta escala

La decisión D8 del SPEC descarta explícitamente estructuras de búsqueda aproximada de vecinos (HNSW, Faiss) a favor de un matmul directo contra la matriz completa de embeddings. Las cuentas:

```
100.000 ítems × 256 dimensiones, en int8  =  25.600.000 bytes  ≈ 25.6 MB
```

Una consulta (un vector de 256 dimensiones) contra esa matriz completa es 25.6M multiplicaciones-suma — trivial para cualquier CPU moderna, del orden de 5-15ms con una implementación vectorizada (numpy con BLAS, o con SIMD int8 si hay soporte). Un índice ANN como HNSW existe para evitar precisamente ese coste lineal, pero solo *paga* cuando el coste lineal es alto — millones de vectores, no cientos de miles. A 100k vectores de 256 dimensiones, recorrer un grafo HNSW (saltos entre nodos, con sus propios *cache misses* al no tener localidad de acceso) no es claramente más rápido que un matmul denso bien vectorizado, y además introduce:

- Tiempo y memoria de construcción del índice.
- Parámetros a ajustar (`M`, `ef_search`) que intercambian velocidad por *recall* — con el riesgo de perder vecinos verdaderamente relevantes.
- Una dependencia extra en el artefacto de despliegue.

El brute-force da **recall exacto (1.0)** — no hay aproximación que ajustar — con cero tiempo de indexación y cero dependencias nuevas. El SPEC fija el criterio de cuándo reconsiderar: por encima de ~1M vectores, donde el coste lineal ya empieza a doler de verdad.

### 8.2 Cuantización int8: qué hace y cuándo falla

La cuantización dinámica convierte los pesos (y, en el caso dinámico, también las activaciones en tiempo de ejecución) de `float32` a `int8`. Dos beneficios simultáneos: 4x menos memoria (dominante en operaciones *memory-bound* como el propio matmul de embeddings del §8.1, donde mover los datos desde RAM suele costar más que la aritmética en sí), y la posibilidad de usar instrucciones enteras vectorizadas de la CPU en vez de las de coma flotante.

Esa segunda parte es la que puede fallar. Las ganancias reales de velocidad con int8 dependen de que la CPU tenga soporte para **VNNI** (*Vector Neural Network Instructions*, parte de AVX-512 en CPUs Intel/AMD relativamente recientes) — instrucciones diseñadas específicamente para acelerar productos punto en enteros de 8 bits. En una CPU sin VNNI, las operaciones int8 o bien se emulan con más pasos (perdiendo la ventaja), o el coste de cuantizar/decuantizar sobre la marcha puede llegar a hacer que int8 sea *más lento* que quedarse en fp32 directamente. Es un problema documentado de ONNX Runtime en hardware antiguo (issue `onnxruntime#6732`, citado en el riesgo R7 del SPEC) — no una posibilidad remota, sino algo que ya se ha observado en despliegues reales.

Por eso la decisión D9 no es "int8 sin más": es int8 **con un fallback fp32 empaquetado**, para no degradar silenciosamente la latencia en un portátil de gama baja o de varios años sin avisar al usuario.

### 8.3 El pipeline ONNX

```
PyTorch (entrenamiento, GPU)
   │  torch.onnx.export
   ▼
grafo ONNX (arquitectura fija, sin dependencia de Python/PyTorch en runtime)
   │  ORT Transformer Optimization Tool
   │  (fusiona operaciones: p.ej. LayerNorm+residual, patrones de atención)
   ▼
cuantización dinámica int8
   │
   ├── model.onnx (int8)              ── principal
   ├── model_fp32.onnx                ── fallback para CPUs sin VNNI
   └── item_emb.int8.npy              ── matriz de embeddings, matmul directo en numpy
   ▼
ONNX Runtime en CPU del usuario, sin GPU, sin entorno Python de entrenamiento
```

La razón de exportar a ONNX en vez de servir directamente con PyTorch (D9) es sobre todo de tamaño y dependencias: un runtime de inferencia PyTorch completo es un artefacto mucho más pesado que ONNX Runtime, y el objetivo de distribución del SPEC (bundle instalable <300MB, §9 criterio 5) no sería alcanzable arrastrando PyTorch entero solo para hacer *forward passes* en producción. La matriz de embeddings se separa a un `.npy` propio porque el paso de vecinos/búsqueda paramétrica (§8.1) no necesita ejecutar el grafo del transformer — es un matmul directo, más simple y más rápido servido fuera del grafo ONNX.

---

## 9. Glosario

- **Token**: unidad mínima de la secuencia que el modelo consume. En un LM de texto, una palabra o subpalabra; aquí, una película o serie del catálogo.
- **Embedding**: vector denso aprendido que representa una entidad discreta (un ítem, una posición, un director...) en un espacio continuo donde la distancia/dirección tiene significado semántico.
- **Softmax**: función que convierte un vector de puntuaciones arbitrarias en una distribución de probabilidad (`exp(x_i)/Σexp(x_j)`), usada tanto para la predicción del siguiente token en un LM de texto como, conceptualmente, para puntuar el siguiente ítem aquí (aunque en la práctica se evita calcularlo completo, §3).
- **Negative sampling**: en vez de comparar contra todo el vocabulario/catálogo, comparar el positivo contra un número reducido de ítems muestreados al azar como negativos — más barato, con el sesgo que corrige gBCE (§3).
- **BCE (binary cross-entropy)**: pérdida que trata cada comparación positivo-vs-negativo como una clasificación binaria independiente.
- **gBCE (generalized BCE)**: variante de BCE con negative sampling que eleva el término positivo a una potencia calibrada (`β`) para corregir el sesgo de sobreconfianza introducido por muestrear pocos negativos (§3.3). De Petrov & Macdonald, RecSys 2023.
- **Máscara causal**: en atención, impide que una posición "vea" posiciones futuras de la secuencia — necesaria para que la predicción del siguiente ítem no haga trampa mirando el propio objetivo.
- **Self-attention / atención multi-cabeza**: mecanismo que permite a cada posición de la secuencia ponderar y combinar información de otras posiciones, mediante proyecciones aprendidas Q (query), K (key), V (value); "multi-cabeza" repite esto en paralelo con proyecciones distintas, permitiendo capturar varios tipos de relación a la vez.
- **LayerNorm (pre-LN vs post-LN)**: normalización de las activaciones de una capa. "Pre-LN" (usado aquí) normaliza antes de entrar al sub-bloque (atención o FFN), dejando limpio el camino residual — más estable de entrenar que la variante "post-LN" original del paper *Attention Is All You Need*.
- **Conexión residual**: `h = h + f(h)` en vez de `h = f(h)` — permite que el gradiente fluya directamente hacia atrás sin atenuarse capa a capa.
- **Weight tying**: reutilizar la misma matriz de parámetros para el embedding de entrada y la proyección de puntuación de salida, en vez de aprender dos matrices separadas (§2.4).
- **Padding**: relleno (aquí, con el índice especial `0`) para que secuencias de longitud variable quepan en un tensor rectangular de longitud fija (`max_len`).
- **NDCG@k (Normalized Discounted Cumulative Gain)**: métrica de ranking que premia acertar el ítem relevante, ponderando más si aparece en las primeras posiciones del top-k que si aparece al final (§5.3).
- **Recall@k**: fracción de casos en los que el ítem relevante aparece dentro de las k primeras posiciones del ranking.
- **Leave-one-out (temporal)**: protocolo de evaluación que reserva el último evento de cada secuencia como test, el penúltimo como validación, y entrena solo con el resto (§5.1).
- **Full-ranking evaluation**: evaluar el ranking del modelo contra el catálogo completo, no contra un subconjunto muestreado de negativos — el estándar defendido frente al *sampled evaluation* (§5.2).
- **Cold-start**: el problema de generar una representación o recomendación útil para un ítem (o usuario) sin historial de interacciones previas.
- **ANN (Approximate Nearest Neighbors)**: estructuras de índice (HNSW, IVF...) para buscar vecinos aproximados en espacios de alta dimensión más rápido que comparando contra todos los puntos — innecesario a la escala de este proyecto (§8.1).
- **Cuantización (int8)**: representar pesos/activaciones con enteros de 8 bits en vez de flotantes de 32, para reducir memoria y, con soporte de hardware adecuado, acelerar el cómputo (§8.2).
- **VNNI**: extensión de instrucciones de CPU (parte de AVX-512) para acelerar productos punto en enteros de 8 bits — condición para que la cuantización int8 realmente acelere la inferencia.
- **ONNX**: formato estándar de grafo de cómputo para modelos entrenados, que permite ejecutarlos con un runtime ligero (ONNX Runtime) sin depender del framework de entrenamiento original.
- **SASRec**: arquitectura de recomendación secuencial basada en un transformer causal (Kang & McAuley, 2018), sin embedding de usuario — la base arquitectónica de este proyecto (§1.2, §2).
- **gSASRec**: variante de SASRec entrenada con la pérdida gBCE en vez de BCE estándar, corrigiendo el sesgo de negative sampling (Petrov & Macdonald, RecSys 2023) — la combinación exacta usada aquí (D1 del SPEC).
- **Timeline mask**: término de la literatura de recsys para la combinación de máscara causal + máscara de padding necesaria quando las secuencias tienen longitud variable y padding — el mecanismo cuyo fallo se documenta en §4.
- **Distribución Zipfiana**: patrón estadístico donde unos pocos elementos concentran la inmensa mayoría de la frecuencia de aparición, y la mayoría de elementos aparecen muy poco — el patrón detrás de la concentración del 98.8% en el corte de 100k títulos (§7.1).
- **Sesión de volcado (rating dump)**: patrón en datasets de valoración explícita donde un usuario puntúa muchos ítems de golpe, con timestamps casi idénticos, independientemente de cuándo consumió cada uno realmente (§5.4).
