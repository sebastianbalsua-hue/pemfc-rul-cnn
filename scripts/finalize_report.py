import json,hashlib,sys,unittest,io
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.paths import ROOT
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    rows=[]
    for p in sorted((ROOT/'results').glob('verified_*/metrics.json')):
        m=json.loads(p.read_text());rows.append({k:m.get(k) for k in ['model','rmse','mae','r2','nasa_score','parameters','trainable_parameters','latency_ms','epochs_completed']})
    table=pd.DataFrame(rows);table.to_csv(ROOT/'results/tables/comparison.csv',index=False)
    lines=['# Resultados de ejecucion local','', 'C-MAPSS FD001: 80 motores de entrenamiento, 20 de validacion, 100 motores oficiales de prueba. Semilla 42. Ventana 30. RUL cap 125 ciclos. Validacion con todas las ventanas de motores reservados; prueba con la ultima ventana por motor.','', '| Modelo | RMSE (ciclos, cap 125) | MAE | R2 | Parametros totales | Latencia ms/muestra | Epocas |','|---|---:|---:|---:|---:|---:|---:|']
    for m in rows:
        latency=m.get('latency_ms');latency='N/D' if latency is None else f'{latency:.3f}'
        lines.append(f"| {m['model']} | {m['rmse']:.3f} | {m['mae']:.3f} | {m['r2']:.3f} | {m['parameters']} | {latency} | {m['epochs_completed']} |")
    lines += ['', 'Las metricas respecto a RUL sin cap tambien se guardan en metrics.json. Las latencias son mediciones locales en CPU y dependen del equipo; no constituyen una prueba estadistica de superioridad. No se ha hecho busqueda de hiperparametros sobre test.', '', 'PHM: no entrenado porque FC1 no alcanza 0.90 V inicial; ver audit.json. ECSIM: caracterizacion parcial desde tabla del articulo; Parquet sin lector local disponible.']
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    (ROOT/'results/logs/tests.txt').write_text(stream.getvalue(),encoding='utf-8')
    manifest=json.loads((ROOT/'docs/original_manifest.json').read_text())
    original=ROOT.with_name('pemfc-rul-cnn');changed=[]
    for item in manifest:
        p=original/item['path']
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']: changed.append(item['path'])
    integrity={'checked_files':len(manifest),'original_files_changed':changed,'original_gitignore_exists':(original/'.gitignore').exists(),'copy_gitignore_exists':(ROOT/'.gitignore').exists()}
    (ROOT/'results/tables/integrity.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
    lines += ['', f'Pruebas: {result.testsRun} ejecutadas, {len(result.failures)} fallos, {len(result.errors)} errores, {len(result.skipped)} omitidas.',f'Integridad original: {len(manifest)} archivos comprobados por SHA-256; {len(changed)} modificados.']
    (ROOT/'docs/RESULTADOS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if rows:
        fig,ax=plt.subplots(figsize=(7,4));ax.bar(table.model,table.rmse,color=['#0e7490','#d97706','#64748b'][:len(table)])
        ax.set(ylabel='RMSE (ciclos, RUL cap 125)',title='C-MAPSS FD001 - prueba oficial, semilla 42');fig.tight_layout();fig.savefig(ROOT/'results/figures/comparison.png',dpi=160);plt.close(fig)
    audit=json.loads((ROOT/'data/processed/phm2014/audit.json').read_text())
    a=pd.read_csv(ROOT/'data/processed/phm2014/FC1_hourly.csv')
    fig,ax=plt.subplots(figsize=(8,4));ax.plot(a.time,a.voltage,label='FC1 horario');ax.axhline(audit['fc1_event']['threshold'],color='#dc2626',label='Umbral 90%')
    ax.set(xlabel='Tiempo (h)',ylabel='Voltaje stack (V)',title='FC1 no cruza el umbral de fallo');ax.legend();fig.tight_layout();fig.savefig(ROOT/'results/figures/phm_censoring.png',dpi=160);plt.close(fig)
    print(stream.getvalue());print(json.dumps(integrity));print(table.to_string(index=False))
    if not result.wasSuccessful() or changed: raise SystemExit(1)
if __name__=='__main__':main()
