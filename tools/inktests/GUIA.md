# Inkling-2-Mini en Sneo: cómo se llegó a 75 tok/s y cómo llegar a 120-150

Modelo: `Inkling-2-Mini-UD-Q4_K_XL.gguf` (arquitectura `inkling2_loop`, 16.4 GB, 26B de parámetros con ~2.7B activos por token). Corre entero en la RTX 3090.

Estado al 2026-11-25:
- Escribe a **~75 tok/s**, unos 13 ms por token.
- Lee prompts largos a ~700 tok/s.
- Los logits coinciden con la referencia numpy (correlación 0.9985).
- Copia literal 6 de 6, tool calls exactas, needle a 22K tokens bien.

Al empezar escribía a 20 tok/s (~47 ms por token). La sección 3 cuenta cómo se pasó de 20 a 75. La sección 5 es el plan para seguir, paso por paso.

---

## 1. Dónde está el código

| Archivo | Qué tiene |
| --- | --- |
| `src/inkrun.h` | Host. `loadInk` (carga y buffers), `inkLayerRun` (una pasada de capa), `inkExperts` (expertos), `inkChunk` (las 48 pasadas; con 16 filas o menos se graban en un grafo de CUDA dentro de la lambda `work`), `forwardInk`, `moveInk` (cortes para reusar prompts) |
| `src/ink.h` | Structs (`InkLayer`, `InkRun`, `InkShape`, `InkBuffers`) y los kernels CUDA de Inkling dentro del string `inkKernelSource`. Se compilan al cargar el modelo; la primera carga después de un cambio tarda ~77 s |
| `src/gpukernels.h`, `src/kernels.h` | Helpers de GPU que ya existen y se pueden usar desde `ink.h`: `unpackGroup(type, ...)`, `groupsPerBlock`, `blockBytes`, `fp4Bytes`, `nvScale`, `hyBlockSum`, `hyHalf`, `toHalf`, `hySigmoid`, `hyQuant`, `typedRows` (el kernel de proyección de pocas filas) |
| `src/hy4.h`, `src/glm5.h`, `src/gpumodel.h` | Helpers de host de otros modelos que Inkling llama: `hyNormRows`, `hyPack`, `hyGatherPack`, `hyRouteFast`, `hyF32Wide`, `glmExpertRows`, `project()`, `plain()`, `go()`. **No se modifican**; si hace falta otra versión, se hace una propia en `ink.h` |

Las 48 pasadas por token salen así: capas 0-5, luego 6-17 tres veces (con compuertas de mensaje entre pasadas), luego 18-23.
- Atención completa en las capas 5, 11, 17 y 23, con conv dinámica en k/v.
- Ventana deslizante de 1024 en las demás.

Tipos de los tensores (todas las capas son iguales salvo lo indicado):

| Tensor | Forma (filas × columnas) | Tipo |
| --- | --- | --- |
| attn_q, attn_gate | 3072 × 2304 | Q8_0 |
| attn_k, attn_v | 1024 × 2304 | Q8_0 |
| attn_rel | 384 × 2304 | Q8_0 |
| attn_output | 2304 × 3072 | Q8_0 |
| attn_sink_proj | 24 × 2304 | BF16 |
| attn_k_conv_proj, attn_v_conv_proj (solo capas full) | 32 × 2304 | BF16 |
| ffn_gate_up_exps | 128 expertos × 2304 × 2304 | NVFP4 en capas 0-19, Q8_0 en 20-23 |
| ffn_down_exps | 128 expertos × 2304 × 1152 | NVFP4 en capas 0-19, Q8_0 en 20-23 |
| ffn_*_exps.scale | 128 floats (escala por experto) | F32, solo capas 0-19 |
| ffn_gate_inp (router), compuertas loop_*_ctrl_*, convs, normas | chicos | F32 |
| output (cabeza) | 201024 × 2304 | Q8_0 (492 MB, se lee entera en cada token) |

Formatos:
- **Q8_0**: bloque de 34 bytes cada 32 valores (escala half de 2 bytes + 32 int8). Los bloques **no** quedan alineados a 4 ni a 16 bytes.
- **NVFP4**: bloque de 36 bytes cada 64 valores (4 escalas E4M3 de 1 byte, una cada 16 valores, + 32 bytes de nibbles). Ver `unpackGroup` con `type == 40` en `gpukernels.h`.

### Lo que lanza una pasada hoy

Pasada normal (ventana deslizante, sin compuertas):

```
inkCopy (solo si empieza un bloque de 6 capas)
inkBlockRes (1 consulta, -> xin)        hyNormRows (xin -> hq)
hyPack(hq)  project q  project rel  project gate  plain sink(BF16)  project k  project v
inkClip q   inkClip rel   inkConv k   inkConv v   inkStore
inkAttendPart  inkAttendMerge
hyGateMulPack   project out   inkConv post   inkAdd (o inkCopy)
inkBlockRes (ffn -> mlpIn)   hyNormRows (mlpIn -> xn)
hyF32Wide (router)   hyRouteFast
inkPairs  hyGatherPack  glmExpertRows(gate_up)  inkAct  glmExpertRows(down)  inkScatter
inkConv ffn   inkAdd
```

- Las pasadas del bucle (2ª y 3ª vuelta de las capas 6-17) agregan: un `inkBlockRes` de mensajes en `inkChunk`, 3 × (`inkMsgDown` + `inkMsgMix`) y 3 normas, y otra compuerta antes de la norma del ffn.
- Las capas full agregan: `inkBlockRes` con 3 consultas (o 3 copias), 3 normas (q/k/v) y las proyecciones BF16 `kProj`/`vProj`.

---

## 2. Cómo probar

Todo está en esta carpeta, `tools/inktests/`. Compila y corre aparte: no toca el `master.exe` de la app.

1. `build.bat` compila el motor en `tools/inktests/build/master.exe`.
2. `run.ps1` lo copia a `build/master_ink.exe`, lo arranca en el puerto 8010 solo con la 3090, carga Inkling, corre un script de preguntas, lo cierra y muestra las líneas importantes del log.
   - Se niega a correr si Sneo o un `master.exe` están abiertos.
   - Parámetros: `-Ask <script>`, `-Context <n>`, `-Tag <nombre del log>`, `-Profile` (perfil por kernel), `-Dump <archivo>` (guarda los logits del prompt), `-Vision`, `-Keep`.
   - El log queda en `logs/<Tag>.log`.

```powershell
& "F:\AI Projects\MiMotor\tools\inktests\build.bat"
& "F:\AI Projects\MiMotor\tools\inktests\run.ps1" -Tag copy -Ask copy.mjs
& "F:\AI Projects\MiMotor\tools\inktests\run.ps1" -Tag prof -Ask ask200.mjs -Profile
```

| Script | Para qué | Tiene que dar |
| --- | --- | --- |
| `copy.mjs` | Copia literal de 6 cadenas con `_`, `:`, JSON, rutas | `exact 6 of 6` |
| `tool.mjs` | Tool calls simples | `read_file` con `{"file":"src/main.js","startLine":10}` y `terminal_list` con `{}` |
| `toolcomplex.mjs` | Código de varias líneas con comillas, listas y objetos anidados | JSON válido e idéntico a lo pedido |
| `long.mjs` | Needle a ~22K tokens (cajas 37, 640, 1290) | 3 OK |
| `ask.mjs` | Paris, haiku, needle a 7K, código | Respuestas correctas |
| `ask200.mjs` | 200 tokens de escritura, para el perfil | Con `-Profile` imprime "kernels over 64 calls" |
| `reuse.mjs` | Dos chats nuevos con el mismo sistema de 3K tokens | El segundo reusa ~3255 tokens y las repeticiones salen iguales |
| `turns.mjs` | Tres vueltas de una conversación con herramientas | Reusa todo menos lo nuevo |
| `multi.mjs` | Render de varias vueltas con herramientas | Con `-Dump` y `ref/rendercmp.py`: idéntico a la plantilla oficial (310 tokens) |

**Python para la referencia** (se arma una sola vez):

```powershell
python -m venv "F:\AI Projects\MiMotor\tools\inktests\ref\venv"
& "F:\AI Projects\MiMotor\tools\inktests\ref\venv\Scripts\python.exe" -m pip install -r "F:\AI Projects\MiMotor\tools\inktests\ref\requirements.txt"
```

Correrlo siempre con `-I`.

**Logits contra la referencia:**

```powershell
& "F:\AI Projects\MiMotor\tools\inktests\run.ps1" -Tag dump -Ask ask1.mjs -Dump "F:\AI Projects\MiMotor\tools\inktests\logs\dump25.bin"
& "F:\AI Projects\MiMotor\tools\inktests\ref\venv\Scripts\python.exe" -I "F:\AI Projects\MiMotor\tools\inktests\ref\inkcmp.py" "F:\AI Projects\MiMotor\tools\inktests\logs\dump25.bin" "F:\AI Projects\MiMotor\tools\inktests\ref\model\tokenizer.json" "F:\AI Projects\MiMotor\tools\inktests\ref\ref25conv.npy"
```

Tiene que dar `corr` ≥ 0.998 y los mismos tokens arriba en `ours` y `ref`.
- `ref/ref25conv.npy` ya está calculado para los 25 ids de `ref/ids25.txt`.
- Para recalcularlo: `python -I ref/inkref.py --ids <contenido de ids25.txt> --out ref/ref25conv.npy`. Tarda ~4 min de CPU; no lo corras a la vez que otra cosa pesada.

**Velocidad real:** se mide sin `-Profile`, con la línea `generated N tokens (X tok/s)` del log. Con el perfil activo no se usa el grafo y todo sale más lento.

---

## 3. Cómo se pasó de 20 a 75 tok/s

### Paso 0: que funcione y que esté bien

- El primer arranque daba error CUDA 700. Las proyecciones BF16 con más de 16 filas van por `plainTensor`, que usa `splitScratch`, y `loadInk` no lo reservaba.
- Logits comparados contra la referencia numpy hasta que coincidieron.

### Paso 1: medir

Se agregó `SNEO_INK_PROFILE`: mide cada kernel con eventos de CUDA e imprime cada 64 tokens. El primer perfil dio ~47 ms por token, y `inkMsgGate` se llevaba 26 (el 55 %).

### Paso 2: el kernel que se comía la mitad (20 → 41 tok/s)

`inkMsgGate` usaba **un solo bloque por fila**.
- Al escribir hay una sola fila, así que trabajaba 1 SM de los 82.
- Leía 2.4 MB de pesos F32, y cada hilo recorría una fila entera de forget/gain: lecturas no coalescidas.

Se partió en dos:
- **`inkMsgDown`**: muchos bloques, un warp por fila de rango, lecturas contiguas.
- **`inkMsgMix`**: un warp por canal, con los lanes recorriendo el rango.

En el mismo cambio, el tramo mínimo de la atención bajó de 64 a 16 claves, lo que da más bloques con contexto corto: `inkAttendPart` bajó de 3.5 a 1.1 ms por token.

### Paso 3: un grafo de CUDA por paso (41 → 74 tok/s)

Son ~2400 lanzamientos por token. `inkChunk` graba toda la pasada en un grafo y lo repite (`replay()`). Para que se pueda grabar:
- Nada puede esperar al host.
- Lo que cambia por token va en memoria de la GPU.
- El alcance de la atención se redondea a potencias de 2. Pasarse es exacto, porque el kernel salta las claves más allá de la posición.

Lección: el grafo solo ayuda cuando los kernels ya son rápidos. Antes del paso 2 daba 23 → 25 tok/s.

### Paso 4: empaquetar una vez (74 → 75.6 tok/s)

q, rel y gate (y k/v cuando comparten entrada) usan la misma entrada cuantizada (`fromPacked` en `inkLayerRun`).

### Arreglos de correctitud

- **Orden de taps de la conv dinámica:** en el peso estático el índice 3 es el token actual; en el dinámico (`dynK`/`dynV`) el índice 0 es el actual. Con otro orden el modelo se come `_`, `:` y `,` al copiar. Lo destapó `copy.mjs`, no el NLL.
- La capacidad de las entradas empaquetadas de los expertos tiene que cubrir `pairs × hidden`.

---

## 4. El método

1. Perfil y elegir el kernel que más pesa.
2. Entender por qué tarda:
   - **¿Bytes leídos contra tiempo?** Por debajo de ~750 GB/s algo está mal repartido.
   - **¿Cuántos bloques lanza?** La 3090 tiene 82 SM; "un bloque por fila" con una fila usa 1 SM.
   - **¿Cuántos lanzamientos?** Un kernel chico cuesta 5-15 µs aunque no haga nada.
3. Arreglar.
4. Verificar (sección 5.1).
5. Medir sin perfil y anotar.

---

## 5. Plan paso a paso hacia 120-150 tok/s

### 5.0 Cuánto se puede ganar

Lo que lee cada token (sale de `ref/bytes.py`):

| Parte | MB por token | Ideal a 750 GB/s | Medido hoy |
| --- | --- | --- | --- |
| Proyecciones de atención (48 pasadas) | 1321 | ~1.8 ms | ~3.9 ms con la cabeza (`rowsDirect8x1` + `plainRows`) |
| Cabeza del vocabulario | 492 | ~0.65 ms | (incluida arriba) |
| 8 expertos por pasada | 1762 | ~2.4 ms | ~3.6 ms (`glmExpertRows`) |
| Resto (router, compuertas F32, normas, convs) | 382 | ~0.5 ms | ~5-6 ms repartidos en decenas de kernels chicos |

El techo está en ~5.5 ms por token. Hoy son ~13.

Perfil por token (sin grafo, antes del empaquetado único):

```
rowsDirect8x1 3.9 ms (289 llamadas)   glmExpertRows 3.6 (96)   inkBlockRes 1.9 (120)
hyNorm 1.3 (154)   inkMsgDown 1.2 (96)   quantizeInput 1.2 (241, ahora menos)
inkAttendPart 1.1 (48)   inkConv 0.9 (192)   inkMsgMix 0.6   hyRouteFast 0.5
hyF32Wide 0.5   inkClip 0.5   inkScatter 0.4   inkAdd 0.4   plainRows 0.35
shortRows 0.3 (1 llamada)   inkAttendMerge 0.3   ...
```

### 5.1 Cómo se trabaja cada paso (siempre igual)

1. **Copia de seguridad** de los dos archivos antes de tocar nada:
   ```powershell
   $b = "F:\AI Projects\MiMotor\tools\inktests\logs\backup_paso1"
   New-Item -ItemType Directory -Force $b | Out-Null
   Copy-Item "F:\AI Projects\MiMotor\src\ink.h", "F:\AI Projects\MiMotor\src\inkrun.h" $b
   ```
   **Nunca** uses `git checkout`, `git stash` ni `git reset` en `F:\AI Projects\MiMotor`. Ese árbol tiene mucho trabajo sin commit (la app, otros modelos) y se perdería. Para deshacer un paso, copia de vuelta los dos archivos del backup.
2. **Implementar** solo ese paso.
3. **Compilar** con `tools/inktests/build.bat` (nunca con el `build.bat` de la raíz mientras la app esté abierta: ese copia al `master.exe` de la app).
4. **Verificar**, en este orden (si algo falla, se arregla o se deshace el paso):
   - `run.ps1 -Ask copy.mjs`: `exact 6 of 6`.
   - `run.ps1 -Ask tool.mjs` y `-Ask toolcomplex.mjs`: argumentos exactos.
   - `run.ps1 -Ask long.mjs -Context 32768`: 3 OK.
   - Logits contra `ref25conv.npy`: `corr` ≥ 0.998.
   - El log no puede decir `inkling2_loop: N rows run without a graph`.
5. **Medir**:
   - `run.ps1 -Ask ask200.mjs` sin perfil: anotar el tok/s.
   - `run.ps1 -Ask ask200.mjs -Profile`: confirmar que el kernel atacado bajó.
6. **Anotar** en `logs/RESULTADOS.md` una fila: paso, tok/s, ms por token (1000 / tok/s), corr, copia, notas.
7. Si mejoró y todo pasa, seguir con el paso siguiente. **No subir nada a GitHub**: eso lo decide el usuario.

### 5.2 Cómo se escribe y se lanza un kernel

- El kernel va dentro del string `inkKernelSource` en `ink.h`, como `extern "C" __global__ void nombre(...)`.
- Puede usar los helpers de la tabla de la sección 1 (todos los fuentes se compilan juntos).
- Se lanza desde `inkrun.h` con:
  ```cpp
  go("nombre", gridX, gridY, blockX, blockY, arg1, arg2, ...);
  ```
- `go` pasa la dirección de cada argumento, así que **los tipos tienen que coincidir exacto** con la firma del kernel:
  - Punteros de GPU como `uint64_t`; `int` como `int`; `float` como `float`.
  - Un `int` donde el kernel espera `u64`, o al revés, corrompe todos los argumentos que siguen.
  - Ojo con expresiones como `n * d`: son `int`; si el kernel espera `u64`, castear.
- Memoria compartida: solo estática y hasta 48 KB (`go` lanza con 0 bytes dinámicos).
- Buffers nuevos: agregar el campo en `InkBuffers` y reservarlo en `loadInk` junto a los demás (`b.gates = grab(C * s.rank * 4);` es un buen lugar). `C` es el bloque de prefill (256 filas); `pairs = C * 8`.
- Todo lo que se lance desde `inkLayerRun` entra solo en el grafo. **No pongas valores por token como argumento**: si algo cambia por token, léelo de `b.positions` u otro buffer de la GPU.
- Cada kernel tiene que andar para `n` de 1 a 256 filas: el mismo código corre en el prefill. Si un kernel nuevo es solo para pocas filas, protegerlo con `if(n <= s.fewRows)` y dejar el camino actual para el resto.

### Paso 1: línea base

No se cambia código.
- Correr `copy.mjs`, `ask200.mjs` (con y sin `-Profile`) y la comparación de logits.
- Crear `logs/RESULTADOS.md` con la primera fila.

Esperado: ~75 tok/s, corr ~0.9985.

### Paso 2: `inkBlockRes` con más hilos (ganancia esperada ~1 ms)

**Por qué:** es un bloque de 256 hilos por fila. Con una fila, cada hilo recorre 9 veces las 2304 columnas sobre 5 estados, y toda la GPU espera a 1 SM. Son 120 llamadas por token, ~16 µs cada una.

**Cómo:**
1. En el kernel, `__shared__ float red[25][8];` pasa a `red[25][32];` (32 warps como máximo).
2. En `inkrun.h`, todos los lanzamientos de `inkBlockRes` (los de `inkLayerRun` y los dos de `inkChunk`: el de mensajes y el final) usan `1024` hilos cuando hay pocas filas y `256` si no:
   ```cpp
   unsigned int resThreads = n <= s.fewRows ? 1024 : 256;
   go("inkBlockRes", spread, 1, resThreads, 1, ...);
   ```
3. Verificar y medir. Si gana poco, pasar a la versión de dos kernels:
   - `inkBlockDots`: grilla (n, 9) que deja las 25 sumas parciales por trozo de 256 columnas en un buffer `[n][9][25]`.
   - `inkBlockMix`: grilla (n, 9) donde cada bloque suma los 9 parciales (barato), calcula los pesos y escribe su trozo.

**Opcional, más ganancia:** fusionar `inkBlockRes` con el `hyNormRows` que le sigue. La norma de la salida se calcula sin otra pasada: si en el primer recorrido también se guardan los productos punto entre los estados (la matriz de Gram 5×5), entonces ‖salida‖² = wᵀ·G·w. En las pasadas del bucle la compuerta va entre medio, así que ahí no se puede.

### Paso 3: las tres compuertas en un lanzamiento (ganancia esperada ~1 ms)

**Por qué:** en cada pasada del bucle se lanzan 3 × (`inkMsgDown` + `inkMsgMix`) + 3 `hyNormRows` = 9 lanzamientos. En las pasadas que no son full, las 3 compuertas leen la misma entrada (`gate(b.xin, 0..2)`).

**Cómo:**
1. `inkMsgDown3`: como `inkMsgDown`, pero recibe 3 entradas, 3 mensajes y 3 pesos `down`. La grilla en y es `3 * rank / 4`, y `blockIdx.y / (rank / 4)` elige la compuerta. Las normas de la entrada se calculan una vez por bloque. Escribe `gates[t][k][rank]` (agrandar `b.gates` a `C * 3 * rank`).
2. `inkMsgMix3`: como `inkMsgMix`, para las 3 compuertas. `go` no tiene dimensión z: lanzar con `gridY = n * 3` y separar con `k = blockIdx.y % 3` (compuerta) y `t = blockIdx.y / 3` (fila).
3. `inkNorm3`: una norma RMS por fila para 3 salidas a la vez (grilla `n * 3`, 256 hilos). Hace lo mismo que `hyNormRows`; compararlo con `hyNormRows` para usar el mismo eps y la misma fórmula.
4. En `inkLayerRun`, reemplazar las 3 llamadas a `gate(...)` y las 3 `hyNormRows` de las pasadas con mensajes (las dos ramas: full y no full).

### Paso 4: convoluciones fusionadas (ganancia esperada ~0.7 ms)

**Por qué:** son 4 `inkConv` y 2 `inkAdd`/`inkCopy` por pasada.

**Cómo:**
1. **k y v juntas:** `inkConv2`, que recibe los dos juegos de punteros (x, w, dyn, history) y usa `c < 1024 ? k : v` con `c` hasta 2048. Un lanzamiento en vez de dos.
2. **post-conv + residual:**
   ```cpp
   extern "C" __global__ void inkConvAdd(const float* x, const float* w, float* history, const float* addend, float* out, int n, int width)
   ```
   Calcula la conv como `inkConv` (sin dinámico) y escribe `out = conv + addend` (o `out = conv` si `addend` es nulo, para el caso `boundary >= 0`). **No pisar `x`**: lo necesita el siguiente token de la misma fila a través de la historia.
3. **ffn-conv + suma:** lo mismo con `b.moe`, `L.ffnConv`, `R.ffnHist` y `addend = b.resid`, `out = next`.
4. **Ojo con el orden de taps:** copiar la lógica de `inkConv` tal cual. Estático: `w0` = 3 tokens atrás … `w3` = actual. Dinámico: `d[0]` = actual … `d[3]` = 3 atrás.

### Paso 5: clips y normas de las capas full (ganancia esperada ~0.4 ms)

1. `inkClip` de q y de rel en un lanzamiento: grilla `n * 24 * 2`, la mitad de los bloques para q (128 hilos activos) y la otra para rel (16 activos). Más simple aún: lanzar con 128 hilos y que el bloque de rel use solo 16.
2. Las 3 normas q/k/v de las capas full: el mismo `inkNorm3` del paso 3, con sus 3 pesos (`L.qNorm`, `L.kNorm`, `L.vNorm`).

### Paso 6: `shortRows` en dos pasadas (ganancia esperada ~0.25 ms)

**Por qué:** el top-64 de los 201K logits usa un solo bloque: 0.3 ms por token.

**Cómo:**
1. `inkTopPart`: grilla de ~128 bloques; cada uno saca el top-64 de su trozo a un buffer `[128][64]` (valor e índice).
2. `inkTopMerge`: 1 bloque que saca el top-64 final de los 8192 candidatos.
3. El formato de salida tiene que ser el mismo que el de `shortRows` (ver cómo lo lee `forwardInk`: `logits.assign(b.hostLogits, b.hostLogits + 2 * shortlist)`, primero valores y después índices). Comparar contra `shortRows` con el mismo vector antes de cambiarlo.

### Paso 7: expertos en 3 lanzamientos especializados (ganancia esperada ~1.2-1.5 ms)

**Por qué:** hoy son 6 lanzamientos (`inkPairs`, `hyGatherPack`, `glmExpertRows` ×2, `inkAct`, `inkScatter`). `glmExpertRows` decide el tipo en tiempo de ejecución (`unpackGroup(type, ...)` con `type` variable).

**Cómo** (solo para `n <= s.fewRows`; el camino de muchas filas queda igual):
1. **Entrada:** con pocas filas, todos los pares de un token usan la misma fila de `b.xn`. En vez de `hyGatherPack` (que copia la fila 8 veces), cuantizar `b.xn` una vez (`hyPack(b.xn, n * d, 0)`) e indexar por `token = par / 8`.
2. **`inkGateUpAct`** (gate_up + activación + cuantizado):
   - Grilla `(1152 / 32, pares)`, 256 hilos. Cada bloque calcula 32 canales `c` de un par; cada warp, 4 canales.
   - Por canal hacen falta dos productos: la fila `c` (gate) y la fila `1152 + c` (up) del experto `ids[par]`.
   - Multiplicar por `guScale[e]` (si no es nulo), aplicar la activación de `inkAct`: `tanh(g/4)*4*sigmoid(g)*tanh(u/25)*25`.
   - Dejar los 32 valores en compartida y que el warp 0 llame a `hyQuant(v, par * 1152 + c0 + lane, ...)` para escribir la entrada cuantizada del down.
   - **Escribir ese cuantizado en `b.hidValues`, `b.hidScales` y `b.hidSums`, no en `packs[0]`.** El `inkAct` actual pisa `packs[0]` porque corre después de que terminó el gate_up. En el kernel fusionado, otros bloques todavía están leyendo `packs[0]` (la entrada `xn`), y pisarlo daría resultados al azar.
   - La entrada `xn` se indexa por token: `first = (par / used) * (2304 / 32)`.
   - Hacerlo **con plantilla o constante de tipo**: `unpackGroup(40, ...)` para NVFP4 y `unpackGroup(8, ...)` para Q8_0, como dos kernels (`inkGateUpAct40`, `inkGateUpAct8`). Así el compilador saca el `switch`.
3. **`inkDownSum`** (down + suma ponderada, sin `inkScatter` ni `b.downOut`):
   ```cpp
   extern "C" __global__ void inkDownSum40(const u8* base, u64 rowBytes, u64 slotBytes, const int* ids, const float* weights, const float* dScale, const i8* packed, const float* scales, const int* sums, float* out, int hidden, int width, int used, GridTables t) {
       int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
       int c = blockIdx.x * 8 + warp, tok = blockIdx.y;
       if(c >= hidden) return;
       int groups = width / 32, perBlock = groupsPerBlock(40), bytes = blockBytes(40);
       float total = 0.0f;
       for(int k = 0; k < used; k++){
           int j = tok * used + k, e = ids[j];
           const u8* line = base + slotBytes * (u64)e + rowBytes * (u64)c;
           u64 first = (u64)j * groups;
           float acc = 0.0f;
           for(int u = lane; u < groups; u += 32){
               int w[8];
               float low, high, shift;
               unpackGroup(40, line + (u / perBlock) * bytes, u % perBlock, w, low, high, shift, t);
               u64 at = first + u;
               const int4* q = (const int4*)(packed + at * 32);
               int4 a = __ldg(q), b = __ldg(q + 1);
               int lowSum = __dp4a(w[3], a.w, __dp4a(w[2], a.z, __dp4a(w[1], a.y, __dp4a(w[0], a.x, 0))));
               int highSum = __dp4a(w[7], b.w, __dp4a(w[6], b.z, __dp4a(w[5], b.y, __dp4a(w[4], b.x, 0))));
               acc += __ldg(scales + at) * (low * (float)lowSum + high * (float)highSum - shift * (float)__ldg(sums + at));
           }
           total += acc * weights[j] * (dScale ? dScale[e] : 1.0f);
       }
       for(int s = 16; s > 0; s >>= 1) total += __shfl_xor_sync(0xffffffff, total, s);
       if(lane == 0) out[(u64)tok * hidden + c] = total;
   }
   ```
   - Grilla `(2304 / 8, n)`, 256 hilos. Escribe `b.moe` directo.
   - Una fila de down tiene 36 grupos de 32 (1152 columnas): con 32 lanes, 4 lanes hacen 2 grupos y el resto 1. Si el perfil muestra poco ancho de banda, que cada medio warp tome un experto distinto (16 lanes × 2 grupos + 4 sobrantes) o que el warp procese 2 expertos a la vez.
   - La variante `inkDownSum8` es igual con `8` en vez de `40`, y para las capas 20-23 `dScale` es nulo.
4. `inkPairs` desaparece: `guScale[e]` y `dScale[e]` se leen adentro de los kernels.
5. En `inkExperts`, la rama `if(n <= s.fewRows)` queda así: `hyPack(b.xn)`, `inkGateUpAct`, `inkDownSum`.
6. Los argumentos salen de la llamada actual a `glmExpertRows`: `L.gateUp.data`, `L.gateUp.stride` (bytes por fila), `L.gateUpBytes` (bytes por experto), `b.ids`, y lo mismo con `L.down`. El tipo está en `L.gateUp.type` (40 o 8).

### Paso 8: proyecciones juntas (ganancia esperada ~1-1.5 ms)

**Por qué:** cada pasada lanza 6-8 proyecciones de una fila, con un warp por fila de pesos. Los bloques Q8_0 de 34 bytes están desalineados, así que no se pueden leer de a 16 bytes.

**Cómo, en dos partes:**

1. **Un lanzamiento por entrada.** Un kernel `inkRowsQ8` que recibe hasta 5 matrices Q8_0 con la misma entrada empaquetada y reparte las filas por tramos:
   - Firma: `const u8* w0..w4, int rows0..rows4, u64 rowBytes, float* out0..out4`, + entrada empaquetada + `cols`.
   - Fila global `r`: si `r < rows0` es de la matriz 0; si no, se resta y se pasa a la siguiente, y así.
   - Grilla `(filas totales / 8)`, 256 hilos (un warp por fila), con el mismo bucle interno que `typedRows` en `gpukernels.h` y `TYPE = 8`.
   - Usos:
     - Pasadas sin compuerta y no full (hq = hk = hv): q, rel, gate, k y v en un lanzamiento (8576 filas).
     - Si no: q, rel y gate (6528 filas) en uno, y k y v en otro.
   - Las BF16 chicas (`sink`, `kProj`, `vProj`) pueden pasar a F32 al cargar (como hace `widened` en `loadInk`) y entrar en el mismo kernel como un tramo F32 leyendo la entrada sin empaquetar. Si complica, dejarlas como están.
2. **Pesos alineados.** Al cargar, reordenar cada matriz Q8_0 de atención en dos arreglos:
   - Escalas (un float por bloque de 32).
   - Valores int8, con cada fila alineada a 16 bytes.
   Así el kernel lee los valores con `int4` (16 bytes) y las escalas aparte. Ocupa casi lo mismo. Se hace en `loadInk` antes de copiar a la GPU, solo para Inkling; las structs `GpuMatrix` de Inkling pueden guardar un puntero extra para las escalas, o hacerse un struct propio.

Hacer primero la parte 1, medir, y después la 2.

### Paso 9 (si sobra tiempo): la cabeza

`output.weight` (201024 × 2304, Q8_0, 492 MB) se lee entero en cada token: ~0.65 ms a 750 GB/s. Con los pesos alineados del paso 8 se acerca a eso. Medir cuánto tarda hoy con el perfil (es un `rowsDirect8x1` con 25128 bloques al final del token).

### 5.3 Cuándo está terminado

- ≥ 120 tok/s en `ask200.mjs` sin perfil.
- `copy.mjs` 6/6, `tool.mjs` y `toolcomplex.mjs` exactos, `long.mjs` 3/3.
- `corr` ≥ 0.998 contra `ref25conv.npy`.
- `reuse.mjs`: el segundo chat reusa ~3255 tokens.
- El prefill no empeora (`long.mjs` lee a ~560 tok/s o más).
- `logs/RESULTADOS.md` con una fila por paso.

---

## 6. Trampas y reglas

- **No cambiar el orden de taps de `inkConv`** (sección 3, arreglos de correctitud).
- **Nada de esperas al host ni valores por token como argumento dentro del grafo.** Si el log dice `inkling2_loop: N rows run without a graph`, la velocidad cae a la mitad.
- El prefill (más de 16 filas) usa sumas atómicas: dos corridas pueden diferir en algún token casi empatado. No es un error. Para comparar, usar los logits contra la referencia y las pruebas exactas.
- `SNEO_INK_ROWS` (bloque de prefill, por defecto 256) se redondea a múltiplos de 64.
- Solo la 3090. `run.ps1` ya pone `CUDA_VISIBLE_DEVICES` con su UUID.
- No probar con la app Sneo abierta, y cerrar los procesos siempre por ruta, nunca por nombre.
- No correr dos trabajos pesados a la vez (la referencia numpy y una compilación, por ejemplo): la PC ya tuvo un error WHEA de CPU.
- **Nunca** `git checkout`, `git stash` ni `git reset` en `F:\AI Projects\MiMotor`: hay mucho trabajo sin commit. Backups a mano (5.1).
- No subir nada a GitHub sin que el usuario lo pida.
- Separar por modelo: solo `ink.h` e `inkrun.h`. En los archivos compartidos, únicamente líneas protegidas con `if(ink)`; optimizar Inkling no puede cambiar cómo corren los otros modelos.
- Estilo del código: sin comentarios, nombres en inglés, `if(` sin espacio, guards con return temprano, pocos helpers.
