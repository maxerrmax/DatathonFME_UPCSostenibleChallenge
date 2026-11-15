import pandas as pd
def split_B(df, val_days=7):
    """Holdout B: els 7 dies ANTERIORS als 7 últims de cada bloc (setmanes diferents de la validació A)."""
    import sys; sys.path.insert(0, "/home/user/DatathonFME_UPCSostenibleChallenge/estandard")
    from data import blocks
    fit, val = [], []
    for b in blocks(df):
        if len(b) < 24 * (2 * val_days + 3):
            fit.append(b); continue
        end = b.index[-1] - pd.Timedelta(days=val_days)
        cut = end - pd.Timedelta(days=val_days) + pd.Timedelta("1h")
        m = (b.index >= cut) & (b.index <= end)
        fit.append(b[~m]); val.append(b[m])
    return pd.concat(fit).sort_index(), pd.concat(val)
