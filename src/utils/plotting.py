def plot_predictions(time,truth,prediction,path,unit='cycles'):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    if truth is not None: ax.plot(time,truth,label='RUL observada')
    ax.plot(time,prediction,label='Prediccion'); ax.set(xlabel='Tiempo / unidad',ylabel=f'RUL ({unit})')
    ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)
