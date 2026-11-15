"""Features + forward numpy de les xarxes de previsió CO2/preu (compartit entrenador <-> agent).

Només numpy. Les features es construeixen a partir de qualsevol objecte amb `.get(key, hora_epoch)` (que retorna
NaN si falta): el `Hist` de l'agent o el `DenseSeries` de l'entrenament. Claus: co2, price, irr, temp, vent.

Causalitat a l'instant t: co2/price fins a t+2 (finestra de 72 h que acaba a t+2); meteo només t-1.
Sortida: 21 valors per als horitzons 3..23 (CO2: log(co2[t+h]/last); preu: (p[t+h]-last)/PRICE_S), last = valor a t+2.
"""
import json
import numpy as np

L = 72                  # llargada de la finestra d'historial (acaba a t+2)
H0, H1 = 3, 23          # horitzons predits
NH = H1 - H0 + 1        # 21
SEQ_C = 3               # canals: propi normalitzat, màscara de faltant, l'altre senyal normalitzat
N_FLAT = 19
N_IN = SEQ_C * L + N_FLAT
EPS = 1e-3
PRICE_S = 0.05
TASKS = ("co2", "price")
MAP = {"co2": "co2_kg_per_kwh", "price": "precio_eur_per_kwh", "irr": "irradiancia_wm2",
       "temp": "temp_ambient_c", "vent": "vent_ms"}


class DenseSeries:
    """Vista densa (NaN on falta) d'un DataFrame horari amb la mateixa interfície `.get` que Hist."""
    def __init__(self, df):
        e = df.index.values.astype("datetime64[h]").astype(np.int64)
        self.e0, self.e1 = int(e.min()), int(e.max())
        n = self.e1 - self.e0 + 1
        self.a = {}
        for k, c in MAP.items():
            arr = np.full(n, np.nan)
            arr[e - self.e0] = df[c].values.astype(float)
            self.a[k] = arr

    def get(self, key, hour):
        i = int(hour) - self.e0
        if i < 0 or i >= len(self.a[key]):
            return np.nan
        return self.a[key][i]


def _win(h, key, t):
    return np.array([h.get(key, t + 2 - (L - 1) + i) for i in range(L)], dtype=float)


def _fill(w):
    ok = np.isfinite(w)
    if not ok.any():
        return None
    idx = np.arange(len(w))
    return np.interp(idx, idx[ok], w[ok])


def _norm(task_key, wf, last):
    if task_key == "co2":
        return np.clip(np.log(np.maximum(wf, EPS) / max(last, EPS)), -3.0, 3.0)
    return np.clip((wf - last) / PRICE_S, -20.0, 20.0)


def _calendar(t):
    days = t // 24
    hod = t % 24
    dow = (days + 3) % 7
    doy = days % 365.25
    a, b, c = 2 * np.pi * hod / 24, 2 * np.pi * dow / 7, 2 * np.pi * doy / 365.25
    oh = np.zeros(7); oh[int(dow)] = 1.0
    return [np.sin(a), np.cos(a), np.sin(b), np.cos(b), np.sin(c), np.cos(c)], oh


def build(h, t, task):
    """Vector d'entrada (N_IN,) float32 per a la tasca 'co2' o 'price' a l'instant t; None si no es pot."""
    t = int(t)
    other = "price" if task == "co2" else "co2"
    w = _win(h, task, t)
    if not np.isfinite(w[-1]):
        return None
    miss = (~np.isfinite(w)).astype(float)
    wf = _fill(w)
    last = float(wf[-1])
    own = _norm(task, wf, last)
    wo = _win(h, other, t)
    wof = _fill(wo)
    if wof is not None and np.isfinite(wo[-1]):
        lo = float(wof[-1])
        oth = _norm(other, wof, lo)
    else:
        lo = np.nan
        oth = np.zeros(L)
    cal, oh = _calendar(t)
    own_lvl = np.log(max(last, EPS)) if task == "co2" else last / PRICE_S
    if np.isfinite(lo):
        oth_lvl = np.log(max(lo, EPS)) if other == "co2" else lo / PRICE_S
    else:
        oth_lvl = 0.0
    irr, tmp, ven = h.get("irr", t - 1), h.get("temp", t - 1), h.get("vent", t - 1)
    mm = float(not (np.isfinite(irr) and np.isfinite(tmp) and np.isfinite(ven)))
    meteo = [irr / 1000.0 if np.isfinite(irr) else 0.0, (tmp - 15.0) / 10.0 if np.isfinite(tmp) else 0.0,
             ven / 5.0 if np.isfinite(ven) else 0.0]
    flat = cal + list(oh) + [own_lvl, oth_lvl] + meteo + [mm]
    x = np.concatenate([own, miss, oth, np.array(flat, dtype=float)]).astype(np.float32)
    assert x.shape[0] == N_IN
    return x


def last_known(h, t, task):
    return float(h.get(task, int(t) + 2))


def targets(h, t, task):
    """Objectiu (21,) a l'espai normalitzat, o None si falta alguna hora."""
    last = last_known(h, t, task)
    y = np.array([h.get(task, int(t) + k) for k in range(H0, H1 + 1)], dtype=float)
    if not (np.isfinite(last) and np.isfinite(y).all()):
        return None
    if task == "co2":
        return np.log(np.maximum(y, EPS) / max(last, EPS))
    return (y - last) / PRICE_S


def decode(task, y, last):
    """Espai normalitzat -> valors físics (kg/kWh o EUR/kWh)."""
    if task == "co2":
        return max(last, EPS) * np.exp(np.clip(y, -3.0, 3.0))
    return last + np.asarray(y) * PRICE_S


# ---------------------------------------------------------------- forward numpy
def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * g + b


def _head(z, P, pre, n_lin, ln):
    for i in range(n_lin):
        z = z @ P[f"{pre}W{i}"] + P[f"{pre}b{i}"]
        if i < n_lin - 1:
            if ln:
                z = _ln(z, P[f"{pre}g{i}"], P[f"{pre}be{i}"])
            z = _gelu(z)
    return z


def forward(arch, P, pre, x):
    """x: (N_IN,) ja estandarditzat. Retorna (21,) estandarditzat."""
    if arch["fam"] == "mlp":
        return _head(x, P, pre, len(arch["hidden"]) + 1, arch["ln"])
    seq = x[:SEQ_C * L].reshape(SEQ_C, L)
    flat = x[SEQ_C * L:]
    hcur = seq
    for i, d in enumerate(arch["dil"]):
        W, b = P[f"{pre}cw{i}"], P[f"{pre}cb{i}"]          # (Cout, Cin, k)
        k = W.shape[2]
        xp = np.pad(hcur, ((0, 0), ((k - 1) * d, 0)))
        out = np.zeros((W.shape[0], L)) + b[:, None]
        for j in range(k):
            out += W[:, :, j] @ xp[:, j * d: j * d + L]
        hcur = _gelu(out)
    z = np.concatenate([hcur[:, -1], hcur.mean(1), flat])
    return _head(z, P, pre, len(arch["hidden"]) + 1, arch["ln"])


class NNBundle:
    """Carrega un nn_<variant>.npz i prediu CO2/preu amb la mitjana de tots els models de la tasca."""
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.P = {k: z[k] for k in z.files}
        self.meta = json.loads(str(self.P["meta"]))

    def predict(self, task, x, last):
        """x: vector de build(); retorna (21,) en unitats físiques (mitjana de models)."""
        m = self.meta["tasks"].get(task)
        if not m:
            return None
        xs = np.clip((x.astype(np.float64) - self.P[f"{task}/xmean"]) / self.P[f"{task}/xstd"], -10.0, 10.0)
        ym, ys = self.P[f"{task}/ymean"], self.P[f"{task}/ystd"]
        outs = []
        for mod in m["models"]:
            y = forward(mod["arch"], self.P, mod["prefix"], xs) * ys + ym
            outs.append(decode(task, y, last))
        return np.mean(outs, axis=0)
