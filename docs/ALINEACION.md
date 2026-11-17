# Alineacion con los documentos suministrados

La carpeta original es de solo lectura durante este trabajo. Esta copia incluye todos sus archivos locales, incluidos datos ignorados por Git, salvo `.gitignore`. No se descarga, publica ni envia nada a GitHub. La copia conserva el historial `.git`; no se realizan commits ni push.

## Arquitecturas oficiales

Principal: CNN 1D causal-dilatada del documento tecnico, seccion 7.1: filtros 32/64/64/128, kernel 3, dilataciones 1/2/4/8, BatchNorm, MaxPooling tras el primer bloque, GlobalAveragePooling, Dense 64/32 con Dropout 0.25/0.2 y salida lineal.

Comparativa: CNN-LSTM de la seccion 7.2, inspirada en la figura del PDF de cambio de enfoque: Conv32/Pool/Conv64/Pool/LSTM64/Dense32/salida. Se conserva el tensor temporal; no se usa Flatten antes de LSTM. LSTM, GRU y regresion lineal quedan como controles adicionales. Atencion/GAU son opcionales y no forman parte del modelo principal implementado.

La menor cantidad de parametros o latencia es una hipotesis a medir, no una propiedad garantizada. BatchNorm agrega estadisticas durante entrenamiento; la garantia implementada es una prediccion al final de una ventana historica, no una salida estrictamente causal por cada paso intermedio durante entrenamiento.

## Decisiones metodologicas y trazabilidad

| Requisito | Implementacion | Fuente |
|---|---|---|
| 3 roles de datos | C-MAPSS: validacion; PHM: destino; ECSIM: caracterizacion | Documento tecnico, secciones 4 y 8 |
| Mediana, Pearson, escalado | Filtro atrasado por unidad; fit solo train; estado serializado | Documento tecnico, fase 2 |
| 14 sensores FD001 | Media de correlaciones absolutas sensor-tiempo por motor train; umbral 0.2 | Documento tecnico, fase 2; umbral es decision de implementacion |
| Ventanas | W=30, paso=1, etiqueta en ultimo instante, sin cruzar motores | Documento tecnico, fase 2 |
| Particion C-MAPSS | 80 motores train, 20 validacion, semilla 42; test oficial separado | Implementacion del principio de no fuga |
| Validacion C-MAPSS | Todas las ventanas de motores reservados; test una ultima ventana por motor | Decision de implementacion |
| Etiquetado PHM | Primer cruce observado de 0.90 V inicial, cap=125 horas | Documento tecnico, fase 2 |
| Censura | Si no hay cruce, no se generan etiquetas exactas | Restriccion matematica de la definicion RUL |
| FC2 | Solo observaciones <=550 h como entradas | Documento tecnico, fase 4 |
| Transferencia | Arquitectura y protocolo; pesos inicializados desde cero | Documento tecnico, seccion 8 |
| Optimizacion | MSE, Adam/SGD configurable, LR=0.001, EarlyStopping=10, reduccion LR | Documento tecnico, fases 4 y 5 |
| Evaluacion | RMSE, MAE, R2, NASA Score, parametros y latencia por muestra | Documento tecnico, seccion 9 |
| Incertidumbre | 100 pasadas MC Dropout, BatchNorm congelada, cuantiles 2.5/97.5 | Documento tecnico, seccion 9; implementacion explicita |

Los cuantiles MC son intervalos empiricos de variabilidad del modelo, no intervalos calibrados de cobertura garantizada ni estimaciones completas de ruido observacional. Se aplican al modelo principal; la baseline fiel al diagrama no incorpora Dropout.

## Resolucion de diferencias entre fuentes

El documento de adaptacion inicial vincula ECSIM al destino; el documento tecnico posterior especifica PHM 2014 como conjunto RUL. Se adopta esa distincion, coherente con el articulo ChemRxiv: sus ensayos son de activacion y caracterizacion, no vidas completas hasta fallo. Las cifras de RMSE o tiempos de fallo descritas como referencia no se convierten en resultados propios ni etiquetas.

## Auditoria PHM y limites

Los CSV originales tienen 143861 filas FC1 y 127369 filas FC2; 14 y 19 timestamps repetidos se consolidan por mediana. Las columnas se conservan con sus unidades originales. Se agregan por horas con intervalos cerrados a la derecha; las ventanas cuentan observaciones agregadas, y no se afirma que intervalos con huecos representen horas consecutivas. No se interpola usando el futuro.

FC1 inicia en 3.317 V: umbral 2.9853 V. El minimo crudo es 3.093 V. No cruza el umbral; la etiqueta exacta de fallo no existe bajo este criterio. El entrenamiento principal PHM queda bloqueado de forma explicita. No se cambia silenciosamente a 95%, no se usa el ultimo registro como fallo y no se reutilizan los tiempos aproximados del texto como verdad.

FC2 local llega a 1020.536306 h, no a 1155 h. Los datos posteriores a 550 h no entran en el pronostico. Cualquier evaluacion retrospectiva necesita declarar el evento real y separar esos datos; este trabajo no reporta metricas PHM inventadas.

## ECSIM

El archivo Parquet se conserva intacto. Falta pyarrow/fastparquet en los entornos locales inspeccionados. No se descargan paquetes por la restriccion de usar lo disponible. Se transcribe la tabla (1-2), pagina 5 del PDF ChemRxiv, para una caracterizacion parcial identificada como datos publicados (RHC=50%, compresion=5%). No se presentan esos seis registros como lectura del Parquet, series de envejecimiento ni curvas EIS completas. El valor 28 mA/cm2 a 0.8 V se conserva tal como aparece en la tabla y requiere revisar el dato original antes de interpretarlo.

## Alcance de la validacion

Una particion y una semilla permiten verificar el flujo, pero no establecen superioridad estadistica. No se completo una busqueda de hiperparametros ni validacion temporal multibloque PHM porque faltan etiquetas bajo el criterio requerido. Fine-tuning entre dominios y ajuste de circuitos EIS quedan fuera del experimento principal. El resultado del programa no equivale a validar RUL en PEMFC mientras FC1 siga censurada.
