# Predicción de Vida Útil Restante (RUL) en Pilas de Combustible PEMFC mediante Adaptación de Arquitecturas Convolucionales CNN

**Repositorio:** `pemfc-rul-cnn`  
**Autores:** Sebastian Andres Baldovino, Santiago Sepúlveda Blanco, Mateo Leiva  
**Programa:** Ingeniería en Ciencia de Datos  
**Tipo de documento:** Documento técnico del proyecto

---

## Tabla de contenido

1. [Introducción](#1-introducción)
2. [Planteamiento del problema](#2-planteamiento-del-problema)
3. [Objetivos](#3-objetivos)
4. [Fuentes de datos](#4-fuentes-de-datos)
5. [Caracterización de los datos](#5-caracterización-de-los-datos)
6. [Metodología](#6-metodología)
7. [Arquitectura del modelo](#7-arquitectura-del-modelo)
8. [Estrategia de transferencia entre dominios](#8-estrategia-de-transferencia-entre-dominios)
9. [Evaluación y métricas](#9-evaluación-y-métricas)
10. [Estructura del repositorio](#10-estructura-del-repositorio)
11. [Limitaciones y consideraciones metodológicas](#11-limitaciones-y-consideraciones-metodológicas)
12. [Referencias](#12-referencias)

---

## 1. Introducción

En el contexto de la transición energética global y la Industria 4.0, la gestión de la salud y el pronóstico operacional de sistemas energéticos avanzados constituyen un pilar crítico para la sostenibilidad, la seguridad operacional y la rentabilidad económica. Las celdas de combustible de membrana de intercambio protónico (PEMFC, *Proton Exchange Membrane Fuel Cell*) han alcanzado una posición de vanguardia en aplicaciones de transporte eléctrico, generación estacionaria y sistemas portátiles debido a su alta densidad de potencia, respuesta dinámica rápida y nula emisión de gases contaminantes en el punto de uso.

Sin embargo, la viabilidad comercial a gran escala de las celdas PEMFC enfrenta un obstáculo determinante: la degradación progresiva, compleja e inevitable de sus componentes electroquímicos esenciales —como la membrana electrolítica de polímero (Nafion) y el ensamblaje electrodo-membrana (MEA)— bajo condiciones operativas dinámicas. Esta degradación se manifiesta en caídas irreversibles de voltaje, pérdidas de eficiencia y eventuales fallas catastróficas del sistema.

Para mitigar estos riesgos, los esquemas de mantenimiento predictivo y de Gestión de Salud y Pronóstico (PHM, *Prognostics and Health Management*) requieren determinar con precisión la Vida Útil Restante (RUL, *Remaining Useful Life*) del equipo. Estimar la RUL permite predecir el horizonte temporal o el número de ciclos operacionales que le restan a la celda antes de rebasar un umbral crítico de falla.

El propósito fundamental de este proyecto es **transferir y adaptar metodologías avanzadas de Aprendizaje Profundo (*Deep Learning*) —específicamente arquitecturas de Redes Neuronales Convolucionales Unidimensionales (1D-CNN) y ConvNets causales-dilatadas optimizadas originalmente para la estimación de RUL en motores turbofán aeroespaciales— al problema de pronóstico de degradación en sistemas PEMFC**. La premisa central se apoya en una coincidencia estructural profunda: aunque los motores turbofán y las celdas de combustible pertenecen a dominios físicos dispares, ambos generan series temporales multivariables complejas a partir de redes de sensores que capturan tendencias de deterioro no lineales, estocásticas y contaminadas con ruido.

---

## 2. Planteamiento del problema

La predicción precisa de la RUL en celdas PEMFC enfrenta severas limitaciones intrínsecas vinculadas a la naturaleza física del sistema y a las restricciones de los modelos computacionales convencionales:

### 2.1 Complejidad y no linealidad del deterioro electroquímico

La degradación en una celda PEMFC es impulsada por fenómenos acoplados de estrés térmico, hídrico, mecánico y electroquímico. Factores como la deshidratación o inundación de la membrana polimérica (Nafion), el envenenamiento del catalizador de platino, la corrosión de las capas de difusión de gas y el cruce de gases (*gas crossover*) generan dinámicas de deterioro altamente no lineales. Estas variables muestran respuestas heterogéneas dependiendo de los ciclos de carga, la humedad relativa y la presión de reactivos ($H_2/O_2$).

### 2.2 Inadecuación de los modelos físicos tradicionales

Los enfoques basados en la física del sistema requieren formular ecuaciones diferenciales complejas que describan el transporte de masa, carga y calor. Dichos modelos exigen simplificaciones e hipótesis idealizadas que degradan su precisión frente a variaciones dinámicas del entorno real, además de requerir un costoso proceso de calibración de parámetros específicos para cada celda.

### 2.3 Inconvenientes computacionales de los modelos recurrentes

Frente a los métodos físicos, los enfoques guiados por datos (*data-driven*) mediante aprendizaje profundo han ganado preeminencia. Históricamente, las Redes Neuronales Recurrentes (RNN), en particular LSTM y GRU, se han empleado para modelar secuencias temporales. Sin embargo, presentan serias desventajas operativas:

- **Elevado número de parámetros:** las puertas de almacenamiento interno multiplican la carga paramétrica.
- **Tiempos de entrenamiento y latencia prolongados:** la naturaleza secuencial del procesamiento paso a paso impide una paralelización eficiente en GPU.
- **Sensibilidad al tamaño de secuencia y degradación del gradiente:** dificultan su despliegue práctico en dispositivos de *edge computing* o sistemas empotrados de control.

### 2.4 Oportunidad de la convolución causal y dilatada

En el campo aeroespacial, estudios sobre el benchmark C-MAPSS de la NASA han demostrado que las arquitecturas puramente convolucionales (basadas en convoluciones causales-dilatadas 1D y escalado compuesto) son capaces de extraer patrones locales de desgaste y dependencias temporales de largo alcance con una fracción de los parámetros y del tiempo de cómputo requeridos por las LSTM. Sin embargo, existe una brecha en la literatura respecto a la adaptación sistemática de estas estructuras convolucionales ligeras para resolver el pronóstico de RUL en series temporales de celdas PEMFC.

### 2.5 Pregunta central de investigación

> **¿En qué medida es posible adaptar una arquitectura de Red Neuronal Convolucional (1D-CNN causal-dilatada), validada experimentalmente en el pronóstico de RUL de motores turbofán, para predecir la degradación no lineal y la vida útil restante en celdas de combustible PEMFC, alcanzando una alta precisión predictiva con una estructura computacionalmente eficiente y compacta?**

---

## 3. Objetivos

### 3.1 Objetivo general

Adaptar y evaluar una arquitectura de Red Neuronal Convolucional 1D causal-dilatada —optimizada originalmente para la predicción de la Vida Útil Restante (RUL) en series temporales multivariables de motores turbofán— al problema de pronóstico de degradación electroquímica en sistemas de celdas de combustible PEMFC, desarrollando un marco de estimación preciso, robusto y computacionalmente eficiente, validado sobre el benchmark IEEE PHM 2014 (FCLAB/FEMTO-ST) y comparado contra arquitecturas recurrentes y CNN-LSTM.

### 3.2 Objetivos específicos

1. **Acondicionar y estructurar las series temporales multivariables** provenientes de los tres datasets de referencia —IEEE PHM 2014 (FCLAB/FEMTO-ST), NASA C-MAPSS y ECSIM (*pem-dataset1*)— ejecutando técnicas de filtrado de ruido, normalización de escala, selección de características mediante análisis de correlación y formateo por ventana deslizante (*sliding window*).

2. **Validar la arquitectura CNN causal-dilatada en el dominio origen (C-MAPSS)**, confirmando su capacidad para extraer patrones de degradación y predecir RUL en series temporales de sensores antes de transferirla al dominio PEMFC.

3. **Diseñar la adaptación arquitectónica de la red convolucional al dominio PEMFC**, integrando capas convolucionales 1D con convoluciones causales y dilatadas, bloques de normalización y mecanismos de atención temporal, permitiendo capturar patrones locales de deterioro y relaciones temporales de largo alcance sin recurrir a capas recurrentes costosas.

4. **Formular e implementar el modelo de etiquetado del objetivo (*Target Labeling*)** para el dataset IEEE PHM 2014, mediante la definición de un umbral de fallo basado en la caída del voltaje de stack, funciones lineales a tramos (*piecewise linear degradation function*) y límites de saturación inicial (*RUL capping*).

5. **Entrenar e hiperparametrizar el modelo predictivo** utilizando algoritmos de optimización (Adam, SGD) y estrategias de regularización (Dropout, Early Stopping) para garantizar la convergencia y evitar el sobreajuste, tanto en C-MAPSS como en PHM 2014.

6. **Validar y comparar el rendimiento del modelo adaptado** frente a arquitecturas de referencia (LSTM, GRU, CNN-LSTM y modelos lineales) utilizando métricas de regresión estandarizadas (RMSE, MAE, $R^2$), la función de puntuación asimétrica de la NASA y evaluando la eficiencia computacional en términos de parámetros entrenables y latencia de inferencia.

7. **Caracterizar electroquímicamente el dominio PEMFC** mediante el dataset ECSIM, analizando curvas de polarización y espectroscopía de impedancia (EIS) para justificar la selección del voltaje de stack como indicador de salud (HI) y validar el pipeline de preprocesamiento sobre datos reales de celdas Nafion 112 y MEA.

---

## 4. Fuentes de datos

El proyecto utiliza tres datasets complementarios, cada uno con un rol metodológico claramente diferenciado:

### 4.1 IEEE PHM 2014 Data Challenge (FCLAB/FEMTO-ST) — *Dataset principal de RUL*

**Rol:** dataset principal para entrenamiento y evaluación del modelo de predicción de RUL en PEMFC.

**Descripción:** benchmark estándar de degradación de pilas PEMFC, ampliamente utilizado en la literatura de PHM. Contiene dos stacks PEMFC de $1\text{ kW}$ con trayectorias de envejecimiento completas ($\sim 1,000\text{ h}$ cada uno) bajo condiciones operativas diferenciadas:

| Característica | FC1 (entrenamiento) | FC2 (prueba/pronóstico) |
|---|---|---|
| Condición operativa | Carga constante $70\text{ A}$ | Carga dinámica con rizado de corriente |
| Celdas | 5 celdas, $100\text{ cm}^2$ | 5 celdas, $100\text{ cm}^2$ |
| Duración | $\sim 800\text{–}1154\text{ h}$ (completo) | $0\text{–}550\text{ h}$ revelado; $551\text{–}1155\text{ h}$ a predecir |
| Variables registradas | Voltaje de stack y celdas, corriente, temperaturas, presiones, flujos, humedad | Idénticas |
| Caracterización | Curvas de polarización + EIS cada $\sim 160\text{ h}$ | Igual, con secuencia distinta |

**Variables:** tiempo de envejecimiento ($\text{h}$), voltajes $U_1\text{–}U_5$ y $U_{tot}$ ($\text{V}$), corriente $I$ ($\text{A}$) y densidad $J$ ($\text{A/cm}^2$), temperaturas de entrada/salida de $H_2$, aire y agua de refrigeración, presiones de $H_2$ y aire, caudales de $H_2$, aire y agua, e higrometría de entrada de aire.

**Enlace:** [IEEE PHM 2014 Data Challenge — FCLAB](https://eng.fclab.fr/ieee-phm-2014-data-challenge/)

---

### 4.2 NASA C-MAPSS — *Dataset de validación arquitectónica*

**Rol:** banco de validación de la arquitectura convolucional en un dominio con etiquetas RUL nativas (*run-to-failure*), antes de transferirla a PEMFC.

**Descripción:** conjunto de datos *Commercial Modular Aero-Propulsion System Simulation* desarrollado por el Centro de Excelencia en Pronósticos de la NASA. Simula el deterioro continuo de motores turbofán bajo cuatro subconjuntos (FD001 a FD004) que combinan modos de falla únicos y múltiples con condiciones operativas variables.

**Variables:** 21 variables de sensores (temperaturas de salida de turbina, presiones de compresor, velocidades de ventilador) y 3 ajustes operacionales.

**Enlace:** [NASA Prognostics Center of Excellence — C-MAPSS](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

---

### 4.3 ECSIM *pem-dataset1* — *Dataset de caracterización electroquímica*

**Rol:** caracterización electroquímica del dominio PEMFC, validación del pipeline de preprocesamiento y justificación física del indicador de salud. **No se utiliza para entrenamiento de RUL** porque no contiene una trayectoria de degradación hasta el fallo.

**Descripción:** repositorio público de código abierto con pruebas experimentales de membrana Nafion 112 y ensamblajes electrodo-membrana (MEA) bajo diversas condiciones operativas y protocolos de activación. Incluye:

- **Curvas de polarización:** voltaje y densidad de potencia en función del flujo de gases $H_2/O_2$ y la humedad relativa.
- **Curvas de impedancia electroquímica (EIS):** para calcular la resistencia del circuito equivalente.
- **Parámetros operacionales:** corriente, voltaje, temperatura, presiones de gas y humedad relativa.

**Enlace:** [ECSIM pem-dataset1 — Kaggle](https://www.kaggle.com/datasets/sepandhaghighi/proton-exchange-membrane-pem-fuel-cell-dataset)

---

## 5. Caracterización de los datos

### 5.1 Resumen comparativo de los tres datasets

| Aspecto | IEEE PHM 2014 | NASA C-MAPSS | ECSIM pem-dataset1 |
|---|---|---|---|
| **Dominio** | PEMFC (degradación real) | Turbofán (degradación simulada) | PEMFC (caracterización) |
| **Variable objetivo** | RUL en horas | RUL en ciclos | No aplica (sin RUL) |
| **Tipo de etiqueta** | Derivada del umbral de voltaje | Nativa (*run-to-failure*) | No tiene |
| **Escala temporal** | $\sim 1,000\text{ h}$ | Cientos de ciclos | Mediciones puntuales |
| **Número de sensores** | $\sim 20+$ variables | 21 sensores + 3 ajustes | $\sim 6\text{–}8$ parámetros |
| **Rol en el proyecto** | Entrenamiento/evaluación RUL | Validación arquitectónica | Caracterización y preprocesamiento |
| **Condiciones operativas** | Carga constante (FC1) y dinámica (FC2) | 6 condiciones (FD001–FD004) | Múltiples ($H_2/O_2$, humedad) |

### 5.2 Caracterización detallada por dataset

#### 5.2.1 IEEE PHM 2014 (FCLAB)

- **Naturaleza:** series temporales continuas de envejecimiento hasta el fallo.
- **Granularidad:** muestreo horario aproximado durante $\sim 1,000\text{ h}$.
- **Fenómenos de degradación:** deshidratación de membrana, envenenamiento del catalizador, corrosión de GDL, *gas crossover*.
- **Indicador de salud (HI) propuesto:** voltaje de stack ($U_{tot}$), complementado opcionalmente con resistencia óhmica estimada vía EIS.
- **Umbral de fallo:** $V_{fail} = 0.90 \cdot V_{inicial}$ ($\sim 800\text{ h}$ en FC1; $\sim 1155\text{ h}$ en FC2).
- **Desafíos:** ruido de sensores, variabilidad operativa, no estacionariedad de la degradación.

#### 5.2.2 NASA C-MAPSS

- **Naturaleza:** series temporales multivariables *run-to-failure*.
- **Granularidad:** un ciclo = un vuelo simulado.
- **Fenómenos de degradación:** desgaste de compresor, turbina, ventilador.
- **Indicador de salud:** RUL etiquetada nativamente.
- **Desafíos:** sensores constantes (controladores de retroalimentación), ruido, modos de falla múltiples.

#### 5.2.3 ECSIM *pem-dataset1*

- **Naturaleza:** mediciones experimentales puntuales bajo condiciones controladas.
- **Granularidad:** curvas de polarización y EIS completas por condición.
- **Fenómenos observados:** efecto de humedad relativa, presiones parciales de $H_2/O_2$ y temperatura sobre el voltaje y la impedancia.
- **Rol:** caracterización electroquímica, análisis de sensibilidad, validación del pipeline de preprocesamiento.
- **Desafíos:** no contiene trayectoria de degradación; requiere tratamiento como datos estáticos de caracterización.

---

## 6. Metodología

El desarrollo metodológico se organiza en seis fases sistemáticas.

### Fase 1 — Adquisición y análisis exploratorio de datos

**Objetivo:** comprender la estructura, calidad y dinámica de los tres datasets antes de cualquier transformación.

**Actividades:**

1. Descarga y verificación de integridad de los tres datasets.
2. Análisis exploratorio (EDA) por dataset:
   - Distribuciones de sensores, valores faltantes, outliers.
   - Series temporales de voltaje, corriente, temperatura y presión.
   - Correlación entre variables y con el tiempo operacional.
3. Documentación en notebooks: `01_eda_phm2014.ipynb`, `02_eda_cmapss.ipynb`, `03_eda_ecsim.ipynb`.

**Entregables:** notebooks de EDA con hallazgos y decisiones de preprocesamiento.

---

### Fase 2 — Preprocesamiento integrado y generación de etiquetas RUL

**Objetivo:** transformar las series temporales crudas en tensores de entrada y etiquetas RUL listas para entrenamiento.

#### 2.1 Selección de sensores por correlación

Se calcula el coeficiente de correlación de Pearson ($r$) entre cada sensor $x$ y el tiempo operacional. Se descartan sensores constantes o con correlación nula (controladores de retroalimentación) y se conservan únicamente las señales con tendencia monótona de desgaste.

**Aplicación:**

- **PHM 2014:** seleccionar $U_{tot}$, $U_1\text{–}U_5$, $I$, temperaturas y presiones con tendencia de degradación.
- **C-MAPSS:** eliminar sensores constantes en FD001; conservar los 14 sensores de tendencia monótona.
- **ECSIM:** conservar voltaje, corriente, temperatura, humedad y presiones para caracterización.

#### 2.2 Filtrado de ruido

Aplicar un **filtro de mediana móvil** (*moving median filter*) sobre las señales seleccionadas. Sustituir cada punto por la mediana de sus vecinos en una ventana temporal definida (p. ej., $k = 5$ o $k = 7$) para atenuar picos de ruido sin distorsionar bordes ni tendencias de degradación.

#### 2.3 Escalado y normalización

Aplicar **normalización Z-Score** o **Min-Max** sobre los datos filtrados para igualar rangos dinámicos entre variables de diferente naturaleza física (voltajes en $\text{V}$ vs. presiones en $\text{kPa}$).

$$
x_{norm} = \frac{x - \mu}{\sigma} \quad \text{(Z-Score)}
$$

$$
x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}} \quad \text{(Min-Max)}
$$

#### 2.4 Generación de etiquetas RUL (modelo lineal a tramos)

**Para PHM 2014:**

1. **Definir el umbral de fallo:** $V_{fail} = 0.90 \cdot V_{inicial}$.
2. **Calcular RUL real:** $RUL(t) = t_{fail} - t$, donde $t_{fail}$ es el tiempo en que $V(t)$ cruza $V_{fail}$.
3. **Aplicar *RUL capping*:** durante la fase saludable inicial, fijar $RUL_{cap}$ constante (p. ej., $125\text{ h}$ o $200\text{ h}$) para evitar que la red aprenda ruido sin tendencia.
4. **Función lineal a tramos:**

$$
RUL_{label}(t) = \begin{cases} RUL_{cap}, & \text{si } RUL(t) \ge RUL_{cap} \\ RUL(t), & \text{si } 0 < RUL(t) < RUL_{cap} \\ 0, & \text{si } t \ge t_{fail} \end{cases}
$$

**Para C-MAPSS:** protocolo estándar con $RUL_{cap} = 125$ ciclos.

#### 2.5 Estructuración en ventanas deslizantes

Transformar las series temporales continuas en muestras secuenciales de tamaño fijo $W$ (p. ej., $W = 25$ o $W = 30$ pasos) con paso de desplazamiento $s$, construyendo tensores de entrada:

$$
X \in \mathbb{R}^{B \times W \times F}
$$

donde $B$ es el tamaño de lote, $W$ la ventana temporal y $F$ el número de características seleccionadas.

**Entregables:** scripts `load_phm2014.py`, `load_cmapss.py`, `load_ecsim.py`, `preprocessing.py`, `labeling.py`; datasets procesados en `data/processed/`.

---

### Fase 3 — Validación arquitectónica en C-MAPSS

**Objetivo:** confirmar que la arquitectura CNN causal-dilatada es capaz de predecir RUL en un dominio con etiquetas nativas antes de transferirla a PEMFC.

**Actividades:**

1. Entrenar la CNN causal-dilatada en FD001 (condición única, falla única) y opcionalmente FD003.
2. Evaluar con RMSE, MAE, $R^2$ y NASA Score.
3. Comparar contra CNN-LSTM y LSTM bajo el mismo protocolo.
4. Documentar curvas de aprendizaje y métricas de referencia.

**Resultado esperado:** $\text{RMSE} \approx 16\text{–}18$ en FD001 (referencia de la literatura para CNN 1D), lo que confirma que la arquitectura extrae patrones de degradación correctamente.

---

### Fase 4 — Adaptación y entrenamiento en PHM 2014

**Objetivo:** entrenar y evaluar la CNN causal-dilatada en el dominio PEMFC con RUL real.

**Actividades:**

1. **Entrenamiento:** FC1 completo (todas las horas hasta el fallo).
2. **Prueba/pronóstico:** FC2 con datos revelados hasta $550\text{ h}$, prediciendo la RUL futura.
3. **Validación cruzada temporal:** división por bloques temporales (no *k-fold* aleatorio) para respetar la causalidad.
4. **Hiperparametrización:** búsqueda de $W$, filtros, dilataciones, tasa de aprendizaje y dropout mediante validación temporal.
5. **Prevención de sobreajuste:** Dropout ($0.2\text{–}0.3$) y Early Stopping monitoreando la pérdida en validación.

**Función objetivo:**

$$
\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

**Optimizador:** Adam con learning rate inicial $\alpha = 0.001$ y decaimiento progresivo.

---

### Fase 5 — Validación, evaluación comparativa y cuantificación de incertidumbre

**Objetivo:** evaluar el rendimiento del modelo adaptado y compararlo contra arquitecturas de referencia.

**Actividades:**

1. **Métricas de regresión:** RMSE, MAE, $R^2$ en horas (PHM 2014) y ciclos (C-MAPSS).
2. **NASA Score asimétrico:** penaliza exponencialmente las predicciones tardías (críticas en mantenimiento predictivo).
3. **Cuantificación de incertidumbre:** Monte Carlo Dropout con 100 pasadas estocásticas en inferencia para generar intervalos de confianza del 95%.
4. **Eficiencia computacional:** número de parámetros entrenables y latencia de inferencia por muestra.
5. **Comparativas mínimas:**
   - CNN causal-dilatada (propuesta).
   - CNN-LSTM baseline.
   - LSTM pura.
   - Persistencia / regresión lineal (baseline trivial).

---

### Fase 6 — Caracterización electroquímica con ECSIM

**Objetivo:** justificar físicamente la selección del voltaje de stack como HI y validar el pipeline de preprocesamiento sobre datos reales de PEMFC.

**Actividades:**

1. Análisis de curvas de polarización: identificar las tres regiones de pérdida (activacional, óhmica, transporte de masa).
2. Análisis de EIS: estimar resistencia óhmica y modelar el circuito equivalente.
3. Análisis de sensibilidad: estudiar el efecto de humedad relativa, presiones parciales de $H_2/O_2$ y temperatura sobre el voltaje y la impedancia.
4. Validar el pipeline de preprocesamiento (filtrado, normalización, selección) sobre las señales de ECSIM.

**Entregable:** notebook `03_eda_ecsim.ipynb` con análisis electroquímico y justificación del HI.

---

## 7. Arquitectura del modelo

### 7.1 Arquitectura principal: CNN 1D causal-dilatada

**Entrada:** tensor `(B, W, F)` — p. ej., `(32, 30, 14)`.

**Flujo:**

```text
Input (B, W, F)
   │
   ├─ Conv1D(32, kernel=3, dilation=1, padding="causal", activation=ReLU)
   ├─ BatchNormalization
   ├─ MaxPooling1D(pool_size=2)
   │
   ├─ Conv1D(64, kernel=3, dilation=2, padding="causal", activation=ReLU)
   ├─ BatchNormalization
   │
   ├─ Conv1D(64, kernel=3, dilation=4, padding="causal", activation=ReLU)
   ├─ BatchNormalization
   │
   ├─ Conv1D(128, kernel=3, dilation=8, padding="causal", activation=ReLU)
   ├─ BatchNormalization
   │
   ├─ [Opcional] Mecanismo de atención temporal o bloque GAU
   │
   ├─ GlobalAveragePooling1D()
   │
   ├─ Dense(64, activation=ReLU)
   ├─ Dropout(0.2 – 0.3)
   │
   ├─ Dense(32, activation=ReLU)
   ├─ Dropout(0.2)
   │
   └─ Dense(1, activation="linear")  → RUL estimada
```

**Hiperparámetros iniciales:**

| Componente | Valor inicial |
|---|---|
| Ventana $W$ | $25\text{–}30$ pasos |
| Filtros | $32 \rightarrow 64 \rightarrow 64 \rightarrow 128$ |
| Kernel | $3$ |
| Dilataciones | $1, 2, 4, 8$ |
| Padding | `causal` |
| Activación | ReLU |
| Pooling | MaxPooling1D o stride |
| Dropout | $0.2\text{–}0.3$ |
| Salida | Dense(1) lineal |
| Loss | MSE |
| Optimizador | Adam, $\text{lr} = 0.001$ |
| Regularización | Early Stopping ($\text{paciencia} = 10$) |

### 7.2 Arquitectura baseline: CNN-LSTM

Inspirada en la arquitectura del paper de turbofán referenciado en la presentación (Kumar et al.). **Nota metodológica:** se corrige el `Flatten` antes de la LSTM presente en la figura original, ya que la LSTM requiere conservar la dimensión temporal `(B, T, C)`.

```text
Input (B, W, F)
  → Conv1D(32, k=3, ReLU)
  → MaxPooling1D(2)
  → Conv1D(64, k=3, ReLU)
  → MaxPooling1D(2)
  → LSTM(64)
  → Dense(32, ReLU)
  → Dense(1, linear)
```

### 7.3 Arquitectura baseline: LSTM pura

```text
Input (B, W, F)
  → LSTM(64, return_sequences=True)
  → LSTM(32)
  → Dense(32, ReLU)
  → Dense(1, linear)
```

---

## 8. Estrategia de transferencia entre dominios

No se transfieren pesos directamente de C-MAPSS a PHM 2014 (los sensores y escalas son radicalmente distintos). En su lugar, se utiliza **transferencia de arquitectura y protocolo** en tres niveles:

### Nivel 1 — Validación arquitectónica en C-MAPSS

Se confirma que la CNN causal-dilatada es capaz de predecir RUL en un dominio con etiquetas nativas. Este nivel responde a la pregunta: *¿funciona la arquitectura en un problema de RUL canónico?*

### Nivel 2 — Reentrenamiento desde cero en PHM 2014

Se usa la misma arquitectura, pero con los sensores y el HI de PEMFC (FC1 $\rightarrow$ FC2). Este nivel responde a la pregunta: *¿es el diseño convolucional agnóstico al dominio físico?*

### Nivel 3 — *Fine-tuning* opcional (experimento adicional)

Preentrenar en C-MAPSS (capas convolucionales) y congelar las primeras capas al entrenar en PHM 2014, permitiendo que el cabezal Dense se adapte a la escala de RUL de PEMFC. Este nivel se documenta como experimento adicional, no como resultado principal, y responde a la pregunta: *¿aporta valor la transferencia de pesos entre dominios dispares?*

### Rol de ECSIM en la transferencia

ECSIM no participa en el entrenamiento de RUL. Su función es:

- Validar el pipeline de preprocesamiento sobre datos electroquímicos reales de PEMFC.
- Caracterizar el dominio: curvas de polarización y EIS que justifican físicamente la elección del voltaje como HI.
- Analizar la sensibilidad del voltaje a humedad, presión y temperatura para argumentar qué sensores incluir en el modelo de RUL.

---

## 9. Evaluación y métricas

### 9.1 Métricas de regresión estándar

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}
$$

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

$$
R^2 = 1 - \frac{\sum_{i=1}^{N} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}
$$

### 9.2 Función de puntuación asimétrica de la NASA

En mantenimiento predictivo real, las predicciones tardías ($\hat{y}_i > y_i$) son gravísimas porque el equipo falla antes de lo previsto, mientras que las predicciones tempranas ($\hat{y}_i < y_i$) solo adelantan la inspección. La función de puntuación de la NASA penaliza exponencialmente los errores tardíos:

$$
S = \sum_{i=1}^{N} s_i, \quad \text{donde } s_i = \begin{cases} e^{-\frac{d_i}{13}} - 1, & \text{si } d_i < 0 \text{ (predicción temprana)} \\ e^{\frac{d_i}{10}} - 1, & \text{si } d_i \ge 0 \text{ (predicción tardía)} \end{cases}
$$

donde $d_i = \hat{y}_i - y_i$.

### 9.3 Cuantificación de incertidumbre

Mediante **Monte Carlo Dropout**, se mantiene activo el Dropout durante la inferencia, ejecutando 100 pasadas estocásticas en cada muestra para calcular la media predictiva y generar un intervalo de confianza del 95% alrededor de la RUL estimada.

### 9.4 Eficiencia computacional

- Número de parámetros entrenables por arquitectura.
- Latencia de inferencia por muestra ($\text{ms}$).
- Uso de memoria ($\text{MB}$).

---

## 10. Estructura del repositorio

```text
pemfc-rul-cnn/
├── data/
│   ├── raw/
│   │   ├── ieee_phm2014/        # FC1_train.csv, FC2_test.csv
│   │   ├── c_mapss/             # train_FD001.txt, test_FD001.txt, RUL_FD001.txt
│   │   └── ecsim/               # pem-dataset1 (caracterización)
│   └── processed/               # ventanas .npz, etiquetas RUL
├── src/
│   ├── data/
│   │   ├── load_phm2014.py
│   │   ├── load_cmapss.py
│   │   ├── load_ecsim.py
│   │   ├── preprocessing.py     # filtrado, normalización, selección
│   │   └── labeling.py          # RUL piecewise + capping
│   ├── models/
│   │   ├── cnn_causal_dilated.py    # arquitectura principal
│   │   ├── cnn_lstm_baseline.py
│   │   └── lstm_baseline.py
│   ├── training/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/
│       └── metrics.py           # RMSE, MAE, R², NASA Score
├── configs/
│   ├── cmapss.yaml
│   ├── phm2014.yaml
│   └── transfer.yaml
├── notebooks/
│   ├── 01_eda_phm2014.ipynb
│   ├── 02_eda_cmapss.ipynb
│   └── 03_eda_ecsim.ipynb
├── results/
│   ├── figures/
│   └── tables/
├── requirements.txt
└── README.md
```

---

## 11. Limitaciones y consideraciones metodológicas

### 11.1 Diferencias de dominio

| Aspecto | C-MAPSS (turbofán) | PHM 2014 (PEMFC) |
|---|---|---|
| Variable objetivo | RUL en ciclos | RUL en horas |
| Escala temporal | Cientos de ciclos | $\sim 1,000\text{ h}$ |
| Sensores | 21 variables de motor | $\sim 20+$ variables electroquímicas |
| Régimen de fallo | Degradación gradual | Degradación electroquímica no lineal |
| Etiquetas RUL | Nativas (*run-to-failure*) | Derivadas del umbral de voltaje |

**Declaración explícita:** la transferencia es **metodológica y arquitectónica, no de pesos**, salvo en el experimento opcional de *fine-tuning* (Nivel 3).

### 11.2 Limitaciones de ECSIM

- No contiene una trayectoria de degradación hasta el fallo.
- No se utiliza para entrenamiento de RUL.
- Su rol es exclusivamente de caracterización electroquímica y validación de preprocesamiento.

### 11.3 Limitaciones del etiquetado RUL en PHM 2014

- El umbral de fallo ($V_{fail} = 0.90 \cdot V_{inicial}$) es una convención del benchmark, no una falla física absoluta.
- La función lineal a tramos con *capping* introduce una simplificación que debe declararse.
- La variabilidad operativa entre FC1 (carga constante) y FC2 (carga dinámica) introduce un desafío adicional de generalización.

### 11.4 Limitaciones de generalización

- Los resultados en FC2 no garantizan generalización a otros stacks PEMFC con diferentes condiciones de operación.
- El modelo no incorpora variables de degradación física interna (p. ej., cruce de gases medido directamente) porque no están disponibles en el dataset.

---

## 12. Referencias

1. Hamidi, S., et al. (2020). *Proton Exchange Membrane (PEM) Fuel Cell Dataset*. ECSIM. [GitHub — ECSIM/pem-dataset1](https://github.com/ECSIM/pem-dataset1)
2. IEEE PHM 2014 Data Challenge. *FCLAB/FEMTO-ST*. [eng.fclab.fr/ieee-phm-2014-data-challenge](https://eng.fclab.fr/ieee-phm-2014-data-challenge/)
3. NASA Prognostics Center of Excellence. *C-MAPSS Dataset*. [NASA PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
4. Kumar, R., Ojha, S., Shelke, A., & Habib, A. *Temporal-sequential modeling for remaining useful life estimation of turbofan engine*.
5. Hong, C. W., Ko, B. H., & Hur, K. (2021). *ConvNet-based remaining useful life prognosis of a turbofan engine*.
6. Zheng, S., et al. (2017). *Long short-term memory network for remaining useful life estimation*. IEEE ICPHM.
7. Khoufi, M., et al. *A hybrid CNN-DNN model for battery remaining useful life (RUL) prediction*.
8. Wikipedia. *Proton-exchange membrane fuel cell*.
9. Wikipedia. *Long short-term memory (LSTM)*.
10. MATLAB & Simulink. *3 formas de estimar la vida útil restante para mantenimiento predictivo*.

---

*Documento generado para el repositorio `pemfc-rul-cnn`. Versión 1.0.*