from pathlib import Path
import pandas as pd

def load_ecsim(path):
    p=Path(path)
    return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)

def characterization(frame):
    """Descriptive characterization only; no synthetic lifetime labels."""
    numeric=frame.select_dtypes('number')
    return {'rows':len(frame),'columns':list(frame.columns),
            'missing':frame.isna().sum().to_dict(),
            'summary':numeric.describe().to_dict(),
            'role':'electrochemical_characterization_only'}
