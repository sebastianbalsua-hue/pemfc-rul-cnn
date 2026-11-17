# Proyecto de Predicción de RUL en Sistemas PEMFC mediante la Adaptación de Arquitecturas Convolucionales (CNN)

## 1. Contextualización y Propósito del Proyecto

En el paradigma de la Industria 4.0 y la transición energética global, la gestión de la salud y el pronóstico operacional de los sistemas energéticos avanzados constituyen un pilar crítico para la sostenibilidad, la seguridad operacional y la rentabilidad económica. Dentro de las tecnologías de generación limpia, las celdas de combustible de membrana de intercambio protónico (PEMFC, por sus siglas en inglés) han alcanzado una posición de vanguardia para aplicaciones de transporte eléctrico, generación estacionaria y sistemas portátiles debido a su alta densidad de potencia, respuesta dinámica rápida y nula emisión de gases contaminantes en el punto de uso.

Sin embargo, la viabilidad comercial a gran escala de las celdas PEMFC enfrenta un obstáculo determinante: la degradación progresiva, compleja e inevitable de sus componentes electroquímicos esenciales —como la membrana electrolítica de polímero y el ensamblaje electrodo-membrana (MEA)— bajo condiciones operativas dinámicas. Esta degradación se manifiesta en caídas irreversibles de voltaje, pérdidas de eficiencia y eventuales fallas catastróficas del sistema.

Para mitigar estos riesgos y evitar interrupciones no programadas, los esquemas de mantenimiento predictivo y de Gestión de Salud y Pronóstico (PHM) requieren determinar con precisión la Vida Útil Restante (RUL, *Remaining Useful Life*) del equipo. Estimar la RUL permite predecir el horizonte temporal o el número de ciclos operacionales que le restan a la celda de combustible antes de rebasar un umbral crítico de falla.

El propósito fundamental de este proyecto es **transferir y adaptar metodologías avanzadas de Aprendizaje Profundo (*Deep Learning*), específicamente arquitecturas de Redes Neuronales Convolucionales Unidimensionales (1D-CNN) y ConvNets optimizadas originalmente para la estimación de RUL en motores turbofán aeroespaciales, e integrarlas al problema de pronóstico de degradación en sistemas PEMFC**. La premisa central se apoya en una coincidencia estructural profunda: aunque los motores turbofán y las celdas de combustible pertenecen a dominios físicos dispares, ambos generan series temporales multivariables complejas a partir de redes de sensores que capturan tendencias de deterioro no lineales, estocásticas y contaminadas con ruido.

---

## 2. Planteamiento del Problema

La predicción precisa de la RUL en celdas de combustible PEMFC enfrenta severas limitaciones intrínsecas vinculadas a la naturaleza física del sistema y a las restricciones de los modelos computacionales convencionales:

* **Complejidad y No Linealidad del Deterioro Electroquímico:**
  La degradación en una celda PEMFC es impulsada por fenómenos acoplados de estrés térmico, hídrico, mecánico y electroquímico. Factores como la deshidratación o inundación de la membrana polimérica (Nafion), el envenenamiento del catalizador de platino, la corrosión de las capas de difusión de gas y el fenómeno de cruce de gases (*gas crossover*) generan dinámicas de deterioro altamente no lineales. Estas variables muestran respuestas heterogéneas dependiendo de los ciclos de carga, la humedad relativa y la presión de reactivos ($H_2/O_2$).

* **Inadecuación de los Modelos Físicos Tradicionales:**
  Los enfoques basados en la física del sistema requieren formular ecuaciones diferenciales complejas que describan el transporte de masa, carga y calor. Dichos modelos exigen simplificaciones e hipótesis idealizadas que degradan su precisión frente a variaciones dinámicas del entorno real, además de requerir un costoso proceso de calibración de parámetros específicos para cada celda.

* **Inconvenientes Computacionales de los Modelos Recurrentes (RNN/LSTM/GRU):**
  Frente a los métodos físicos, los enfoques guiados por datos (*data-driven*) mediante aprendizaje profundo han ganado preeminencia. Históricamente, las Redes Neuronales Recurrentes (RNN), en particular los modelos de Memoria a Corto y Largo Plazo (LSTM) y las Unidades Recurrentes Con Puertas (GRU), se han empleado para modelar secuencias temporales. Sin embargo, estas redes recurrentes presentan serias desventajas operativas:
  * **Elevado número de parámetros y complejidad estructural:** Las puertas de almacenamiento interno (*gates*) multiplican la carga paramétrica.
  * **Tiempos de entrenamiento y latencia prolongados:** La naturaleza secuencial del procesamiento paso a paso impide una paralelizacion eficiente en hardware acelerador (GPU).
  * **Sensibilidad al tamaño de secuencia y degradación del gradiente:** Dificultan su despliegue práctico en dispositivos de monitoreo en el borde (*edge computing*) o en sistemas empotrados de control de celdas de combustible.

* **La Oportunidad de la Convolución Causal y Dilatada:**
  En el campo aeroespacial, estudios sobre el benchmark C-MAPSS de la NASA han demostrado que las arquitecturas puramente convolucionales (basadas en convoluciones causal-dilatadas 1D y escalado compuesto) son capaces de extraer patrones locales de desgaste y dependencias temporales de largo alcance con una fracción de los parámetros y del tiempo de cómputo requeridos por las LSTM. Sin embargo, existe una brecha en la literatura respecto a la adaptación sistemática de estas estructuras convolucionales ligeras y de amplio campo receptivo para resolver el pronóstico de RUL en series temporales de celdas de combustible PEMFC.

**Pregunta Central de Investigación:**
¿En qué medida es posible adaptar una arquitectura de Red Neuronal Convolucional (1D-CNN / ConvNet), validada experimentalmente en el pronóstico de RUL de motores turbofán, para predecir la degradación no lineal y la vida útil restante en celdas de combustible PEMFC, alcanzando una alta precisión predictiva con una estructura computacionalmente eficiente y compacta?

---

## 3. Formulación de Objetivos

### 3.1 Objetivo General
Adaptar y evaluar una arquitectura de Red Neuronal Convolucional (1D-CNN / ConvNet) —optimizada originalmente para la predicción de la Vida Útil Restante (RUL) en series temporales multivariables de motores turbofán— al problema de pronóstico de degradación electroquímica en sistemas de celdas de combustible PEMFC, desarrollando un marco de estimación preciso, robusto y computacionalmente eficiente.

### 3.2 Objetivos Específicos
1. **Acondicionar y estructurar las series temporales multivariables** provenientes de los datasets de referencia de motores turbofán (NASA C-MAPSS) y de celdas de combustible PEMFC (ECSIM / pem-dataset1), ejecutando técnicas de filtrado de ruido, normalización de escala, selección de características mediante análisis de correlación y formateo por ventana deslizante (*sliding window*).
2. **Diseñar la adaptación arquitectónica de la red convolucional**, integrando capas convolucionales 1D, bloques de convolución causal y dilatada (*dilated causal convolutions*) o mecanismos de filtrado jerárquico, permitiendo capturar patrones locales de deterioro y relaciones temporales de largo alcance sin recurrir a capas recurrentes costosas.
3. **Formular e implementar el modelo de degradación del objetivo (Target Labeling)** mediante funciones lineales a tramos (*piecewise linear degradation function*) y límites de saturación inicial (*RUL capping*), determinando de forma automática el punto de inflexión donde se inicia el deterioro observable del sistema.
4. **Entrenar e hiperparametrizar el modelo predictivo** utilizando algoritmos de optimización (Adam, SGD) y estrategias de regularización (Dropout, Early Stopping) para garantizar la convergencia y evitar el sobreajuste.
5. **Validar y comparar el rendimiento del modelo adaptado** frente a arquitecturas de referencia (LSTM, GRU, CNN-LSTM y modelos lineales) utilizando métricas de regresión estandarizadas (RMSE, MAE, $R^2$) y evaluando la eficiencia computacional en términos de parámetros entrenables y latencia de inferencia.

---

## 4. Origen de las Ideas y Antecedentes del Dominio

### 4.1 La Evolución del Pronóstico en Motores Turbofán (Benchmark C-MAPSS)
El conjunto de datos C-MAPSS (*Commercial Modular Aero-Propulsion System Simulation*), desarrollado por el Centro de Excelencia en Pronósticos de la NASA, se ha consolidado como el estándar de prueba (*benchmark*) primario para validar modelos de mantenimiento predictivo. Este entorno simula el deterioro continuo de motores turbofán bajo cuatro subconjuntos (FD001 a FD004) que combinan modos de falla únicos y múltiples con condiciones operativas variables.

La literatura científica en torno a C-MAPSS ha atravesado cuatro etapas evolutivas clave:
* **Métodos Estadísticos y Machine Learning Tradicional:** Modelos iniciales basados en Regresión por Vectores de Soporte (SVR), ARIMA y redes someras requerían una compleja extracción manual de características y mostraban baja capacidad de generalización frente a dinámicas no lineales.
* **Surgimiento de Redes Recurrentes (LSTM y Bi-LSTM):** La introducción de redes LSTM permitió aprender las dependencias temporales de las series temporales de sensores. Investigaciones avanzadas como las de Zheng et al. y Asif et al. demostraron la efectividad de combinar redes LSTM profundas con filtros de mediana móvil y análisis de correlación de Pearson para seleccionar los sensores con tendencia monótona de degradación.
* **Modelos Híbridos CNN-LSTM:** Para mejorar la representación espacial de múltiples sensores de forma simultánea con el aprendizaje de secuencia, diversos autores como Kumar et al., Jayasinghe et al. y Yao et al. combinaron capas convolucionales 1D iniciales (que actúan como extractoras de características locales) seguidas de capas LSTM apiladas.
* **Arquitecturas Puramente Convolucionales (ConvNet / Dilated Causal CNN):** Investigaciones destacadas, como la desarrollada por Hong, Ko y Hur, cuestionaron la necesidad de incluir capas recurrentes. Al emplear convoluciones Causales y Dilatadas 1D combinadas con el principio de escalado compuesto (*compound scaling*) de EfficientNet y bloques de atención, demostraron que una red exclusivamente convolucional puede expandir exponencialmente su campo receptivo (*receptive field*). Esto permite capturar dependencias de larga secuencia reduciendo drásticamente la cantidad de parámetros y acelerando el tiempo de cómputo en comparación con modelos CNN-LSTM o LSTM profundos.

### 4.2 La Brecha en Sistemas Electroquímicos PEMFC y el Justificante de Transferencia
Paralelamente, la investigación en pronóstico de salud para celdas PEMFC y baterías electroquímicas ha seguido un camino análogo. En el campo de las baterías de ion-litio, trabajos como los de Khoufi et al. han demostrado la capacidad de los modelos híbridos CNN-DNN combinados con selección de características mediante optimización por enjambre de partículas (BPSO) para estimar la RUL con coeficientes de determinación ($R^2$) superiores al 99%.

En el ámbito específico de las celdas de combustible PEMFC, los repositorios de datos abiertos como ECSIM (*pem-dataset1*) han proporcionado mediciones experimentales de caracterización electroquímica, incluyendo curvas de polarización (variación de voltaje y densidad de potencia según flujos de $H_2/O_2$ y humedad) y curvas de espectroscopia de impedancia electroquímica (EIS).

**Justificación de la Transferencia de Conocimiento (Domain Transfer):**
A pesar del éxito comprobado de las arquitecturas convolucionales ligeras (ConvNets causales dilatadas) en motores turbofán, su aplicación directa sobre la degradación de celdas PEMFC permanece subexplorada. Ambas plataformas comparten características fundamentales:
* **Naturaleza de los Datos:** Ambas generan secuencias temporales multivariables de sensores (temperatura, presión, voltaje, corriente, flujos).
* **Patrón de Salud de Dos Etapas:** En ambos sistemas, el equipo opera inicialmente en un estado saludable con variaciones estables, seguido de un punto de inflexión a partir del cual se desata una tendencia de degradación monótona no lineal hasta alcanzar el fallo.
* **Necesidad de Eficiencia en el Borde:** Tanto la aviación como la movilidad eléctrica basada en hidrógeno requieren que los algoritmos de pronóstico se ejecuten en tiempo real dentro de unidades de control con capacidad de cómputo limitada.

---

## 5. Metodología Detallada y Paso a Paso del Proyecto

El desarrollo metodológico del proyecto se organiza en cinco fases sistemáticas:

1. **Fase 1: Adquisición y Análisis Exploratorio de Datos:**
   * **Dataset de Origen (C-MAPSS de la NASA):** Contiene 709 trayectorias de entrenamiento *run-to-failure* y 707 trayectorias de prueba de motores turbofán, registrando 21 variables de sensores (temperaturas de salida de turbina, presiones de compresor, velocidades de ventilador) y 3 ajustes operacionales.
   * **Dataset de Destino (ECSIM pem-dataset1):** Consiste en datos experimentales sobre celdas PEMFC con membrana Nafion 112 y MEA bajo múltiples condiciones operativas. Registra parámetros electroquímicos esenciales: voltaje de celda, densidad de corriente, temperatura de operación, presión parcial de gases de alimentación ($H_2/O_2$) y porcentaje de humedad relativa.

2. **Fase 2: Preprocesamiento Integrado de Señales y Generación de Etiquetas:**
   * **Selección de Sensores por Correlación:** Se calcula el coeficiente de correlación de Pearson ($r$) entre cada variable de sensor $x$ y el tiempo operacional. Se descartan aquellos sensores constantes o con correlación nula (causados por controladores de retroalimentación interna), conservando únicamente las señales con tendencia monótona de desgaste.
   * **Filtrado de Ruido:** Se aplica un filtro de mediana móvil (*moving median filter*) sobre las señales seleccionadas. Al sustituir cada punto por la mediana de sus vecinos en una ventana temporal definida, se atenúan los picos de ruido sin distorsionar los bordes ni las tendencias de degradación.
   * **Escalado y Normalización:** Se aplica normalización Z-Score o Min-Max sobre los datos filtrados para igualar los rangos dinámicos entre variables de diferente naturaleza física (e.g., voltajes en voltios vs. presiones en kPa).
   * **Generación de Etiquetas RUL (Piecewise Linear Degradation Model):** En las etapas iniciales de operación, el sistema no presenta degradación medible. Se implementa una función lineal a tramos (*piecewise linear function*) con límite de saturación (*capping*), donde la RUL se mantiene constante en un valor inicial (e.g., $RUL_{cap} = 125$ ciclos) durante la fase saludable, y decae de forma estrictamente lineal hacia cero a partir del punto de inflexión identificado automáticamente.
   * **Estructuración en Ventanas Deslizantes (*Sliding Window*):** Las series temporales continuas se transforman en muestras secuenciales de tamaño fijo $W$ (e.g., $W = 25$ o $W = 30$ pasos de tiempo) con un paso de desplazamiento $s$, construyendo tensores de entrada con formato $(B, W, F)$, donde $B$ es el tamaño de lote y $F$ el número de características seleccionadas.

3. **Fase 3: Adaptación e Innovación Arquitectónica de la Red Convolucional:**
   El diseño del modelo predictivo sustituye las complejas capas recurrentes por un marco convolucional 1D altamente eficiente:
   * **Bloque de Entrada y Extracción Local:** Filtros convolucionales unidimensionales (`Conv1D`) con activación ReLU extraen patrones espaciales y relaciones locales de corto alcance entre sensores adyacentes.
   * **Convoluciones Causales Dilatadas (*Dilated Causal Convolutions*):** Para capturar el contexto temporal de largo alcance sin perder resolución ni incrementar desmedidamente la cantidad de parámetros, se introducen factores de dilatación crecientes ($D = 1, 2, 4, 8, \dots$). La condición de causalidad garantiza que la predicción en el tiempo $t$ dependa exclusivamente de las observaciones pasadas e inmediatas, evitando la fuga de información futura.
   * **Bloques de Activación con Puertas (GAU) y Mecanismos de Atención:** Se integran unidades de activación tipo *gated* (combinando funciones $\tanh$ y sigmoides) o capas de atención temporal para ponderar dinámicamente los instantes temporales y sensores que exhiben mayor firmeza sintomática de falla.
   * **Red de Regresión Final (Dense Head):** Las representaciones convolucionales extraídas se reducen mediante capas de promediado global (*Global Average Pooling 1D*) o aplanado (*Flatten*), conectándose a capas completamente conectadas (Dense/DNN) con regularización Dropout antes de emitir la predicción escalar continua de RUL en la capa de salida lineal.

4. **Fase 4: Entrenamiento, Optimización e Hiperparametrización:**
   * **Función Objetivo:** Minimización del Error Cuadrático Medio (MSE) sobre el conjunto de entrenamiento:
     $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$
     donde $y_i$ es la RUL real etiquetada y $\hat{y}_i$ la RUL predicha por la red.
   * **Estrategia de Optimización:** Comparación sistemática de algoritmos de optimización (Adam frente a Descenso de Gradiente Estocástico, SGD) aplicando un esquema de decaimiento de la tasa de aprendizaje (*learning rate schedule*), iniciando en $\alpha = 0.001$ y reduciéndose progresivamente para estabilizar la convergencia alrededor de los mínimos locales.
   * **Prevención del Sobreajuste:** Aplicación de capas Dropout (con tasas entre $0.2$ y $0.3$) e implementación de detención temprana (*Early Stopping*) monitoreando la pérdida en el conjunto de validación con una paciencia fija.

5. **Fase 5: Validación, Evaluación Comparativa y Cuantificación de Incertidumbre:**
   * **Métricas de Regresión Estándar:** Error Cuadrático Medio ($\text{RMSE} = \sqrt{\text{MSE}}$), Error Absoluto Medio ($\text{MAE} = \frac{1}{N} \sum |y_i - \hat{y}_i|$) y Coeficiente de Determinación ($R^2$).
   * **Puntuación Asimétrica de la NASA (Score Function):** En el mantenimiento predictivo real, las predicciones tardías ($\hat{y}_i > y_i$) son gravísimas porque el equipo falla antes de lo previsto, mientras que las predicciones tempranas ($\hat{y}_i < y_i$) solo adelantan la inspección. La función de puntuación de la NASA penaliza exponencialmente los errores tardíos:
     $$S = \sum_{i=1}^{N} s_i, \quad \text{donde } s_i = \begin{cases} e^{-\frac{d_i}{13}} - 1, & \text{si } d_i < 0 \text{ (predicción temprana)} \\ e^{\frac{d_i}{10}} - 1, & \text{si } d_i \ge 0 \text{ (predicción tardía)} \end{cases}$$
     donde $d_i = \hat{y}_i - y_i$.
   * **Cuantificación de Incertidumbre via Monte Carlo Dropout:** Se mantiene activo el Dropout durante la fase de inferencia, ejecutando $100$ pasadas estocásticas en cada muestra para calcular la media predictiva y generar un intervalo de confianza del $95\%$ alrededor de la RUL estimada.

---

## 6. Características y Conceptos Clave para la Comprensión del Proyecto

Para facilitar la asimilación y el estudio del proyecto, a continuación se detallan los conceptos fundamentales distribuidos en sus tres dominios del conocimiento:

### 6.1 Conceptos Fundamentales de Mantenimiento Predictivo y PHM

| Concepto | Definición y Relevancia en el Proyecto |
| :--- | :--- |
| **Mantenimiento Predictivo (PdM) / CBM** | Estrategia de mantenimiento basada en el monitoreo continuo de la condición real del equipo (*Condition-Based Maintenance*). Reemplaza al mantenimiento reactivo (reparar tras la falla) y al preventivo estático (reparar por calendario), optimizando los costos operativos. |
| **Vida Útil Restante (RUL)** | Intervalo temporal, número de ciclos o capacidad operacional remanente desde el instante de observación actual hasta que el sistema alcanza el umbral de falla funcional. Es la variable objetivo del modelo predictivo. |
| **Estado de Salud (SOH)** | Métrica porcentual ($0\%$ a $100\%$) que cuantifica la salud actual del equipo en términos de degradación de capacidad o incremento de resistencia interna en comparación con sus condiciones nominales de fábrica. |
| **Modelos de Similitud, Supervivencia y Degradación** | Enfoques analíticos de RUL: los de *supervivencia* usan distribuciones de probabilidad históricas; los de *similitud* comparan perfiles de sensores actuales con trayectorias pasadas hasta el fallo; los de *degradación* proyectan cuándo un indicador cruzará un umbral crítico. |
| **Límite de Saturación de RUL (*RUL Capping*)** | Técnica de preprocesamiento de etiquetas que fija un límite máximo constante a la RUL durante la fase inicial de operación saludable, evitando que la red intente aprender patrones de degradación inexistentes al inicio del ciclo de vida. |

### 6.2 Conceptos Electrotécnicos y Electroquímicos de PEMFC

| Concepto | Definición y Relevancia en el Proyecto |
| :--- | :--- |
| **Celda PEMFC** | Dispositivo electroquímico de conversión directa que combina Hidrógeno ($H_2$) en el ánodo u Oxígeno ($O_2$/Aire) en el cátodo a través de una membrana polimérica electrolítica, produciendo energía eléctrica, calor y agua limpia. |
| **Ensamblaje Electrodo-Membrana (MEA)** | Componente núcleo de la celda PEMFC compuesto por la membrana de intercambio protónico (generalmente Nafion), las capas de catalizador de platino y las capas de difusión de gas (GDL). Es la estructura principal sujeta a degradación. |
| **Nafion 112** | Polímero fluorado ampliamente utilizado como membrana electrolítica debido a su alta conductividad protónica cuando está hidratado. Su deshidratación o degradación mecánica degrada el rendimiento de la celda. |
| **Curva de Polarización** | Gráfica fundamental que representa la densidad de corriente ($A/cm^2$) frente al voltaje de salida de la celda ($V$). Muestra las tres regiones de pérdida: activacional, óhmica y de transporte de masa. |
| **Espectroscopia de Impedancia (EIS)** | Técnica de diagnóstico electroquímico que aplica señales sinusoidales de corriente para medir la resistencia interna de la celda y modelar el circuito equivalente de degradación. |

### 6.3 Conceptos de Aprendizaje Profundo y Procesamiento Convolucional

| Concepto | Definición y Relevancia en el Proyecto |
| :--- | :--- |
| **1D-CNN (Convolución 1D)** | Operación matemática que desliza filtros unidimensionales a lo largo de la dimensión temporal de una serie temporal multivariable, extrayendo características y patrones de correlación locales entre sensores. |
| **Convolución Causal Dilatada** | Tipo de convolución donde el filtro aplica un paso de salto (*dilation rate*) entre sus elementos, expandiendo exponencialmente el campo receptivo de la red a lo largo del tiempo sin incrementar la cantidad de parámetros ni violar el principio de causalidad. |
| **Campo Receptivo (*Receptive Field*)** | Rango temporal total de la secuencia de entrada que influye en la predicción de una neurona en las capas profundas. Un campo receptivo amplio permite capturar tendencias de degradación de largo plazo. |
| **Modelo Lineal a Tramos (*Piecewise Linear Function*)** | Función matemática empleada para definir la trayectoria del objetivo RUL, dividiendo el ciclo de vida en una meseta inicial constante y una rampa descendente lineal hasta la falla. |
| **Escalado Compuesto (*Compound Scaling*)** | Estrategia de optimización de redes convolucionales (proveniente de EfficientNet) que escala equilibradamente la profundidad, el ancho de canales y la resolución temporal de la red para maximizar la precisión con el menor costo computacional. |
