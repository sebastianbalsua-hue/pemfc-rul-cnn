# PEM-RUL-CNN — copia local de trabajo

Modelo principal: **CNN 1D causal-dilatada**. Comparativa: **CNN-LSTM**, conservando la dimension temporal antes de LSTM. Implementacion basada exclusivamente en los documentos y datos locales suministrados.


## Ejecucion

Desde esta carpeta, usar un Python con NumPy, pandas y PyYAML; TensorFlow se requiere para redes neuronales. El entorno local disponible es `C:/Users/sebas/Formacion_uis/deep_learning/.venv/Scripts/python.exe`.

```powershell
$py = 'C:/Users/sebas/Formacion_uis/deep_learning/.venv/Scripts/python.exe'
& $py -m unittest discover -s tests -v
& $py scripts/preprocess_all.py
& $py scripts/run_experiments.py --models cnn cnn_lstm linear --epochs 100 --tag nueva_comparacion
& $py scripts/train_phm2014.py
```

El ultimo comando se detiene intencionalmente: **FC1 no cruza 90% del voltaje inicial**. Su RUL exacta no se puede obtener bajo la definicion proporcionada. No se fabrican etiquetas ni resultados PEMFC.

Las carpetas de resultados deben tener nombres nuevos; no se sobrescriben experimentos. Los archivos `configs/cmapss.yaml` y `configs/phm2014.yaml` controlan procesamiento/entrenamiento. Las configuraciones de arquitectura documentan los valores fijos implementados en `src/models`; no son un buscador automatico de arquitecturas.

## Resultados y trazabilidad

- `docs/ALINEACION.md`: decisiones, requisitos, limitaciones y discrepancias entre documentos.
- `docs/RESULTADOS.md`: resultados medidos de la comparacion local.
- `docs/original_manifest.json`: hashes de la copia inicial.
- `docs/sources/`: los cuatro documentos suministrados.
- `data/processed/cmapss/`: ventanas y estado de preprocesamiento ajustado solo en train.
- `data/processed/phm2014/audit.json`: evidencia de censura y corte FC2.
- `results/verified_*/`: modelos, historia, predicciones, metricas y preprocesamiento.
- `results/tables/`: resumen y verificacion de integridad.
- `notebooks/`: exploracion reproducible y lectura de resultados locales.

ECSIM se usa exclusivamente para caracterizacion. El Parquet requiere un lector no disponible; se incluye un analisis parcial de la tabla del articulo, claramente identificado. Las pruebas usan casos pequenos de control; los experimentos C-MAPSS usan los datos locales reales.
