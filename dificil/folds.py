"""Validació multi-plec del Mode Difícil.

Plec k (1..K): la k-èsima setmana comptant des del final de cada bloc mensual del train
(k=1 és la validació de l'equip: els darrers 7 dies de cada bloc, igual que el test).
Els models del plec k s'entrenen amb tot el train excepte aquestes setmanes:
    python dificil/train_v7.py f2        -> dificil/agent_hard/models_f27.pkl
"""
import pandas as pd


def _blocks(df):
    gid = (df.index.to_series().diff() != pd.Timedelta("1h")).cumsum()
    return [g for _, g in df.groupby(gid)]


def split_fold(df, k, days=7):
    fit, val = [], []
    for b in _blocks(df):
        if len(b) < 24 * (days * k + 1):                 # cal com a mínim 1 dia de context abans
            fit.append(b)
            continue
        hi = b.index[-1] - pd.Timedelta(days=days * (k - 1)) + pd.Timedelta("1h")
        lo = hi - pd.Timedelta(days=days)
        m = (b.index >= lo) & (b.index < hi)
        fit.append(b[~m]); val.append(b[m])
    return pd.concat(fit), pd.concat(val)
