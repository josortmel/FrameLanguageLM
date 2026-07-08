# TensorBoard — guía de lectura

Qué significa cada scalar que ves ahora mismo en `logs/tb`. Un apartado por métrica.

## `train/loss`

Es la gBCE (generalized Binary Cross-Entropy): para cada paso, el modelo puntúa el ítem real siguiente contra un puñado de negativos (256 aquí) y la loss castiga si el negativo puntúa más alto que el real.

Lo que ves en TensorBoard NO es la loss media de la época. Es una EMA (exponential moving average, 0.9 anterior + 0.1 nuevo) actualizada cada 100 steps globales. Por eso el trazo es más suave que la loss cruda por batch, pero reacciona rápido a cambios reales.

Forma sana: bajada rápida en los primeros steps → meseta suave que sigue bajando despacio. Como aprender a reconocer caras: al principio confundes todo, luego afinas.

Señales de alarma:
- **Sube y no baja** varios cientos de steps seguidos → learning rate demasiado alto, o negativos mal muestreados.
- **NaN** → casi siempre explosión numérica (gradientes, softmax con logits enormes). Si aparece, para el training, no esperes a que se arregle solo.
- **Plana desde el step 0** → el modelo no está aprendiendo nada. Revisa que los datos lleguen bien (batch no vacío, vocabulario correcto), no asumas que "ya converge".

## `valid/ndcg10`

Mide calidad del ranking en el top-10, con descuento logarítmico por posición: acertar en el puesto 1 vale más que acertar en el puesto 9.

En cristiano: para cada usuario, escondes la última película que vio y le pides al modelo un top-10 de recomendaciones. `ndcg10` te dice si esa película acertada aparece, y si aparece, si está arriba (mucho valor) o casi cayéndose del top-10 (poco valor). Si no aparece, vale 0 para ese usuario.

Es la media de esto sobre todos los usuarios de validación.

Rango esperado: empieza cerca de 0 (modelo sin entrenar, aleatorio). Con SASRec en MovieLens, la literatura reporta convergencia sobre ~0.10-0.15. Si ves 0.20+ sospecha algo raro (fuga de datos, split mal hecho); si se estanca en 0.02-0.03 tras muchas épocas, algo no está funcionando.

## `valid/recall10` y `valid/recall50`

Más simple que ndcg: "¿en qué % de usuarios la película real cayó dentro del top-10 (o top-50)?" — sin importar si fue el puesto 1 o el 10, cuenta igual.

Relación con ndcg10: recall10 es el techo que ndcg10 nunca puede superar (si no está en el top-10, ndcg10 tampoco cuenta ese acierto). Si recall10 sube pero ndcg10 se queda plano, el modelo está metiendo el ítem correcto en el top-10 pero en mal puesto (ej. siempre puesto 9, nunca puesto 1-2) — mejora "que aparezca", no "que aparezca bien colocado".

recall50 es más generoso, útil para ver si el modelo al menos tiene la idea general aunque no afine el top-10.

## `time/epoch_seconds` y `time/eta_minutes`

`epoch_seconds`: tiempo real que tardó esa época, medido con reloj de pared.

`eta_minutes`: épocas restantes estimadas × media de los últimos 5 `epoch_seconds`. "Épocas restantes" es el mínimo entre:
- lo que falta hasta `--epochs` (tope duro, 200 aquí salvo que Pepe lo cambiara), y
- lo que falta hasta que salte el early stopping (patience=5: si `valid/ndcg10` lleva 5 épocas sin mejorar sobre su mejor marca, para).

Por eso el ETA puede oscilar: cada vez que `ndcg10` mejora, el contador de patience se resetea y el ETA puede subir de golpe (más margen antes de parar). No es un bug, es que la meta móvil real es "cuándo para", no la época 200 — en la práctica casi siempre para mucho antes por early stopping.

## Cómo leer la sesión de hoy

**Early stopping**: el training para solo cuando `valid/ndcg10` lleva 5 épocas seguidas sin superar su mejor valor histórico. El checkpoint guardado (`sasrec_baseline.pt`) es siempre el de la MEJOR época, no la última — aunque pare en la época 40, el checkpoint puede ser de la época 35.

**Curva sana**: `train/loss` baja y `valid/ndcg10` sube, más o menos en paralelo, ambas con meseta suave al final.

**Overfitting**: `train/loss` sigue bajando (o incluso cae en picado) mientras `valid/ndcg10` se estanca o empieza a caer. El modelo está memorizando secuencias de entrenamiento en vez de aprender patrones que generalizan a usuarios de validación. Si ves esto, el early stopping ya te protege — el checkpoint bueno quedó guardado antes de que empezara la caída.

**La métrica que manda**: `valid/ndcg10`. Es la única que decide qué checkpoint se guarda y cuándo para el entrenamiento. `loss`, `recall10/50` y los tiempos son contexto para diagnosticar — pero si tienes que mirar un solo número mientras esperas, es ese.
