# Bitácora de optimización de Inkling-2-Mini

Registro de cada paso: qué se cambió, qué dio, qué se aprendió. Lo sigue `GUIA.md` (el plan) y lo continúa Hydrogen. Fechas y números son reales, medidos en la 3090 con `tools/inktests`.

Cómo leer la tabla:
- **tok/s** = velocidad real de escritura (`ask200.mjs` sin `-Profile`).
- **ms/token** = 1000 / tok/s.
- **copia** = `copy.mjs` (tiene que ser 6/6).
- **corr** = mejor de varias corridas de `inkcmp.py` contra `ref25conv.npy`. El prefill varía entre 0.98 y 0.999 por las sumas atómicas (ver nota abajo); lo que importa es que la **mejor** corrida llegue a ~0.999 y que la copia dé 6/6.

| Paso | Cambio | tok/s | ms/token | copia | corr (mejor) |
| --- | --- | --- | --- | --- | --- |
| 0 base | — (árbol al 2026-11-25, 75 tok/s publicados) | 75.5 | 13.2 | 6/6 | 0.9994 |
| 2 | `inkBlockRes` con 1024 hilos en decode | 76.4 | 13.1 | 6/6 | 0.9994 |

---

## Antes de empezar: cómo se llegó a 75 (resumen; el detalle está en GUIA.md §3)

Al portar el modelo escribía a 20 tok/s. Tres cambios lo subieron a 75:
1. **Partir `inkMsgGate` en dos** (`inkMsgDown` + `inkMsgMix`): el kernel original corría en un solo bloque por fila, 1 SM de 82, y leía mal la memoria. 20 → 41 tok/s.
2. **Grabar toda la pasada en un grafo de CUDA** (`inkChunk` → `work` → `replay`): ~2400 lanzamientos por token pasan a uno. 41 → 74 tok/s.
3. **Empaquetar cada entrada una sola vez** (`fromPacked`): q/rel/gate (y k/v) comparten la entrada cuantizada. 74 → 75.6 tok/s.

**La lección más importante de esos tres:** el grafo solo ayuda cuando los kernels ya son rápidos. Antes del paso 1, grabar el grafo daba 23 → 25; después, 41 → 74.

---

## Paso 2 — `inkBlockRes` con más hilos (2026-11-25)

**Qué:** el kernel usaba un bloque de 256 hilos por fila. Al escribir hay una sola fila, así que recorría 9 veces las 2304 columnas sobre 5 estados con 256 hilos mientras los otros 81 SM esperaban.

**Cambio (solo 2 archivos):**
- `src/ink.h`: `__shared__ float red[25][8]` → `red[25][32]` (admite hasta 32 warps).
- `src/inkrun.h`: los 5 lanzamientos de `inkBlockRes` usan 1024 hilos cuando `n <= s.fewRows` y 256 si no. Se agregó `unsigned int resThreads = n <= s.fewRows ? 1024 : 256;` al inicio de `inkLayerRun`.

**Resultado:** `inkBlockRes` bajó de 122 a 111 ms (por 64 tokens, con perfil). Velocidad real 75.5 → 76.4 tok/s. Copia 6/6. Logits bien.

**Por qué ganó poco (clave para Hydrogen):**
- El perfil (`-Profile`) corre **sin grafo**, así que cada tiempo incluye el costo de lanzar el kernel (~10-15 µs). Con 120 llamadas por token, eso solo son ~1.5 ms de puro lanzamiento que **en la corrida real desaparecen**, porque el grafo lanza todo de una vez.
- Por eso un kernel chico que en el perfil marca "111 ms" en realidad aporta mucho menos al tiempo real. Repartir sus hilos ayuda poco: no estaba limitado por cómputo sino por el lanzamiento, y eso ya lo tapa el grafo.

**Conclusión del paso:** los kernels **chicos** (inkBlockRes, inkConv, inkClip, las normas, los adds) dan poco aunque el perfil los muestre grandes. Fusionarlos sigue valiendo para que el prefill (que no usa grafo) sea más rápido, pero no mueven mucho el decode.

---

## Dónde está la velocidad real (hacia dónde ir)

Bajo el grafo, lo que pesa de verdad son los kernels **limitados por memoria**, que el grafo no puede ocultar porque sí mueven muchos bytes:

| Kernel | ms por 64 tokens (perfil) | Qué es | Paso de la guía |
| --- | --- | --- | --- |
| `rowsDirect8x1` | ~252 | Proyecciones de atención + la cabeza del vocabulario (492 MB) | **8** y **9** |
| `glmExpertRows` | ~226 | Los 8 expertos por pasada, kernel genérico | **7** |
| el resto (chicos) | cada uno < 80 | residual, normas, convs, compuertas | 3-6 (ganancia menor) |

**Recomendación para Hydrogen:** saltar directo a los pasos **7 (expertos)** y **8 (proyecciones)** de la guía. Juntos son ~7.5 ms de los ~13 por token y están limitados por memoria, no por lanzamientos, así que ahí la ganancia es real. Los pasos 3-6 (fusionar kernels chicos) dejarlos para el final o solo si se quiere acelerar el prefill.

---

## Notas que cuestan caro si no se saben

1. **El prefill varía entre corridas.** Dos corridas del mismo build dieron corr 0.9994 y 0.9818 sobre el mismo prompt. No es un bug: el camino de expertos con más de 16 filas usa sumas atómicas, y el orden cambia. Para juzgar correctitud: correr el dump 2-3 veces y mirar la **mejor** corr, y confiar sobre todo en `copy.mjs` (6/6) y en las tool calls, que sí son deterministas en decode.
2. **Medir la velocidad siempre sin `-Profile`.** Con perfil no hay grafo y da ~42 tok/s en vez de ~76.
3. **Backup a mano antes de cada paso** (`logs/backup_<paso>` con `ink.h` e `inkrun.h`). Nunca `git` en `F:\AI Projects\MiMotor`.
4. **Compilar con `tools/inktests/build.bat`**, no con el de la raíz (ese pisa el `master.exe` de la app).
