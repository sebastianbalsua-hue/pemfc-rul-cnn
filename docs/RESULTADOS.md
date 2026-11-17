# Resultados de ejecucion local

C-MAPSS FD001: 80 motores de entrenamiento, 20 de validacion, 100 motores oficiales de prueba. Semilla 42. Ventana 30. RUL cap 125 ciclos. Validacion con todas las ventanas de motores reservados; prueba con la ultima ventana por motor.

| Modelo | RMSE (ciclos, cap 125) | MAE | R2 | Parametros totales | Latencia ms/muestra | Epocas |
|---|---:|---:|---:|---:|---:|---:|
| cnn | 17.514 | 13.324 | 0.809 | 56161 | 16.578 | 22 |
| cnn_lstm | 15.076 | 11.625 | 0.858 | 42721 | 22.813 | 20 |
| linear | 20.873 | 16.239 | 0.729 | 15 | N/D | 0 |

Las metricas respecto a RUL sin cap tambien se guardan en metrics.json. Las latencias son mediciones locales en CPU y dependen del equipo; no constituyen una prueba estadistica de superioridad. No se ha hecho busqueda de hiperparametros sobre test.

PHM: no entrenado porque FC1 no alcanza 0.90 V inicial; ver audit.json. ECSIM: caracterizacion parcial desde tabla del articulo; Parquet sin lector local disponible.

Pruebas: 17 ejecutadas, 0 fallos, 0 errores, 0 omitidas.
Integridad original: 75 archivos comprobados por SHA-256; 0 modificados.
