"""Ajuste VE Peste Branca - núcleo de cálculo em Python.

Port direto do núcleo da página (mesmas regras, mesmos nomes de conceitos).
Lê o mapa e os logs do Master Injection, calcula a correção do mapa principal
de combustível (#F01..#F16) e grava um novo .csv trocando só essas linhas.

Uso (linha de comando):
    python ajuste_ve.py --map "CLAUDE 6.csv" --log log_12-11.csv \
        --prev "log_CLAUDE_5.csv" "CLAUDE 5.csv" \
        --saida "MAPA C2 PESTE BRANCA CLAUDE 7.csv"

Logs anteriores (--prev) recebem pesos 0,5, 0,25, 0,125... na ordem em que
aparecem (do mais novo para o mais velho), e cada um usa o mapa que estava
no carro quando foi gravado.
"""
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass

import numpy as np

N = 16
NEED = ['RPM', 'MAP', 'Lambda 1', 'Lambda Target', 'Lambda Corr', 'Lambda Loop', 'CLT', 'ACC %']
EXTRA = ['Inj. Utiliz.', 'Knock', 'IAT', 'KM/H', 'Ign. Adv.']

DEFAULTS = dict(
    lag=10, cltC=82, tol=0.01, bigTol=0.03, minFull=10, minBig=4,
    minHalf=1.5, maxChange=0.15, strength=1.0,
)


class FormatError(ValueError):
    """Arquivo que não é do formato esperado."""


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return math.nan


# ---------- mapa ----------

@dataclass
class MapFile:
    text: str
    eol: str
    fuel: np.ndarray    # fuel[i, j]: i = eixo MAP (linha), j = eixo RPM (coluna)
    rpm_ax: np.ndarray
    map_ax: np.ndarray
    name: str = ''


def parse_map(text: str, name: str = '') -> MapFile:
    eol = '\r\n' if '\r\n' in text else '\n'
    fuel: list = [None] * N
    rpm_ax = None
    map_ax = None
    for ln in re.split(r'\r?\n', text):
        parts = ln.strip().split(';')
        tag = parts[0]
        nums = [_num(x) for x in parts[1:17]]
        m = re.fullmatch(r'#F(\d\d)', tag)
        if m and 1 <= int(m.group(1)) <= 16:
            fuel[int(m.group(1)) - 1] = nums
        elif tag == '#I20':
            rpm_ax = nums
        elif tag == '#I21':
            map_ax = nums

    for i, row in enumerate(fuel):
        if row is None or len(row) != N or any(not math.isfinite(v) for v in row):
            raise FormatError(
                'O arquivo não parece um mapa do Master Injection: '
                f'falta a linha #F{i + 1:02d} do mapa principal.')
    if rpm_ax is None or map_ax is None or len(rpm_ax) != N or len(map_ax) != N:
        raise FormatError('O arquivo não tem os eixos de RPM e MAP (#I20 e #I21).')
    for k in range(1, N):
        if not (rpm_ax[k] > rpm_ax[k - 1]) or not (map_ax[k] > map_ax[k - 1]):
            raise FormatError('Os eixos de RPM ou MAP do arquivo não estão em ordem crescente.')

    return MapFile(
        text=text, eol=eol, name=name,
        fuel=np.array(fuel, dtype=float),
        rpm_ax=np.array(rpm_ax, dtype=float),
        map_ax=np.array(map_ax, dtype=float),
    )


def _fmt(v: float) -> str:
    v = float(v)
    return str(int(v)) if v.is_integer() else repr(v)


def write_map(m: MapFile, new_fuel: np.ndarray) -> str:
    """Troca só as linhas #F01..#F16; o resto do arquivo fica idêntico."""
    out = []
    for ln in re.split(r'\r?\n', m.text):
        mm = re.match(r'#F(\d\d);', ln.strip())
        if mm and 1 <= int(mm.group(1)) <= 16:
            i = int(mm.group(1)) - 1
            out.append('#F%02d;' % (i + 1) + ';'.join(_fmt(v) for v in new_fuel[i]))
        else:
            out.append(ln)
    return m.eol.join(out)


# ---------- log ----------

@dataclass
class LogData:
    n: int
    cols: dict

    def col(self, name: str) -> np.ndarray:
        return self.cols[name]


def parse_log(text: str) -> LogData:
    lines = re.split(r'\r?\n', text)
    header = None
    for ln in lines:
        if ln.startswith('Mess 1;'):
            header = ln.split(';')
            break
    if header is None:
        raise FormatError('Não achei o cabeçalho do log (linha que começa com "Mess 1;"). '
                          'Confira se é um log do Master Injection.')
    idx = {c: (header.index(c) if c in header else -1) for c in NEED + EXTRA}
    miss = [c for c in NEED if idx[c] < 0]
    if miss:
        raise FormatError('O log não tem a(s) coluna(s): ' + ', '.join(miss) + '.')

    rows = {c: [] for c in NEED + EXTRA}
    for ln in lines:
        if not ln.startswith('#D01;'):
            continue
        p = ln.split(';')
        for c in NEED + EXTRA:
            k = idx[c]
            rows[c].append(_num(p[k]) if (0 <= k < len(p) and p[k] != '') else math.nan)

    n = len(rows['RPM'])
    if n < 50:
        raise FormatError(f'O log tem poucos registros ({n}).')
    cols = {c: np.array(rows[c], dtype=float) for c in NEED + EXTRA}
    return LogData(n=n, cols=cols)


def load_text(path: str) -> str:
    # latin-1 preserva os bytes do arquivo do Master Injection
    with open(path, encoding='latin-1', newline='') as f:
        return f.read()


# ---------- cálculo ----------

def fwd_max_min(a: np.ndarray, lag: int):
    """Máx e mín em cada janela t..t+lag (a janela encurta no fim do log)."""
    n = len(a)
    hi = np.full(n, -np.inf)
    lo = np.full(n, np.inf)
    for k in range(lag + 1):
        if k >= n:
            break
        s = a[k:]
        valid = ~np.isnan(s)
        h = hi[:n - k]
        l = lo[:n - k]
        np.maximum(h, np.where(valid, s, -np.inf), out=h)
        np.minimum(l, np.where(valid, s, np.inf), out=l)
    mx = np.where(np.isfinite(hi), hi, np.nan)
    mn = np.where(np.isfinite(lo), lo, np.nan)
    return mx, mn


def cell_of(v, ax):
    """Índice da célula inferior e peso (0..1) na interpolação bilinear."""
    v = np.asarray(v, dtype=float)
    c = np.searchsorted(ax, v, side='left')
    i = np.clip(c - 1, 0, len(ax) - 2)
    t = (v - ax[i]) / (ax[i + 1] - ax[i])
    t = np.clip(t, 0.0, 1.0)
    return i, t


@dataclass
class Analysis:
    W: np.ndarray
    S: np.ndarray
    C: np.ndarray
    used: int
    ok: np.ndarray
    f: np.ndarray


def analyze_log(log: LogData, rpm_ax, map_ax, opt: dict) -> Analysis:
    lag = opt['lag']
    n = log.n
    L = log.col
    tgt = L('Lambda Target')
    corr = L('Lambda Corr')
    rpm = L('RPM')
    mp = L('MAP')
    clt = L('CLT')

    tmx, tmn = fwd_max_min(tgt, lag)
    mmx, mmn = fwd_max_min(mp, lag)
    amx, amn = fwd_max_min(L('ACC %'), lag)
    rmx, rmn = fwd_max_min(rpm, lag)
    _, lmn = fwd_max_min(L('Lambda Loop'), lag)

    lam = np.full(n, np.nan)
    if n > lag:
        lam[:n - lag] = L('Lambda 1')[lag:]          # sonda 10 registros depois

    with np.errstate(invalid='ignore', divide='ignore'):
        ok = ((tmx - tmn) <= 10) & ((mmx - mmn) <= 5) & ((amx - amn) <= 60) & \
             ((rmx - rmn) <= 600) & (lmn == 1) & (clt >= opt['cltC'] + 273) & \
             (lam < 1074) & (lam > 650) & (rpm > 500)
        f = (corr / 1000) * (lam / tgt)
        ok &= np.isfinite(f)

    W = np.zeros((N, N))
    S = np.zeros((N, N))
    C = np.zeros((N, N))
    idx = np.nonzero(ok)[0]
    if idx.size:
        ri, rt = cell_of(rpm[idx], rpm_ax)
        mi, mt = cell_of(mp[idx], map_ax)
        ff = f[idx]
        cc = corr[idx]
        for dm, wm in ((0, 1 - mt), (1, mt)):
            for dr, wr in ((0, 1 - rt), (1, rt)):
                w = wm * wr
                keep = w > 0
                ii = mi[keep] + dm
                jj = ri[keep] + dr
                np.add.at(W, (ii, jj), w[keep])
                np.add.at(S, (ii, jj), (w * ff)[keep])
                np.add.at(C, (ii, jj), (w * cc)[keep])
    return Analysis(W=W, S=S, C=C, used=int(idx.size), ok=ok, f=f)


def combine_from(cur: np.ndarray, items: list, opt: dict) -> dict:
    """items: [{'a': Analysis, 'base': mapa usado no log, 'weight': peso}]."""
    cur = np.asarray(cur, dtype=float)
    SW = np.zeros((N, N))
    SF = np.zeros((N, N))
    for e in items:
        a = e['a']
        base = np.asarray(e['base'], dtype=float)
        wgt = e['weight']
        mask = a.W > 0
        m = np.divide(a.S, a.W, out=np.zeros((N, N)), where=mask)
        fr = m * base / cur
        SW += np.where(mask, a.W * wgt, 0.0)
        SF += np.where(mask, fr * a.W * wgt, 0.0)

    tol, big, maxCh = opt['tol'], opt['bigTol'], opt['maxChange']
    k = opt['strength']
    NEW = cur.copy()
    mean = np.full((N, N), np.nan)
    status = [['' for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            W = SW[i, j]
            if not W > 0:
                status[i][j] = 'sem'
                continue
            mu = SF[i, j] / W
            mean[i, j] = mu
            err = mu - 1
            full = (W >= opt['minFull'] and abs(err) > tol) or (W >= opt['minBig'] and abs(err) > big)
            half = (not full) and W >= opt['minHalf'] and abs(err) > big
            fac = 1.0
            if full:
                fac = min(max(mu, 1 - maxCh), 1 + maxCh)
            elif half:
                fac = 1 + min(max(err, -maxCh), maxCh) * 0.5
            fac = 1 + (fac - 1) * k
            NEW[i, j] = round(cur[i, j] * fac)        # meio para o par, como o Python e a página
            if NEW[i, j] != cur[i, j]:
                status[i][j] = 'meia' if half else 'ok'
            else:
                status[i][j] = ('dentro' if abs(err) <= tol else 'borda') if W >= opt['minBig'] else 'pouco'
    return dict(NEW=NEW, mean=mean, W=SW, per=[e['a'] for e in items], status=status, opt=opt)


def combine(cur, rpm_ax, map_ax, entries: list, opt: dict) -> dict:
    """entries: [{'log': LogData, 'base': mapa do log, 'weight': peso}]."""
    items = [{'a': analyze_log(e['log'], rpm_ax, map_ax, opt), 'base': e['base'], 'weight': e['weight']}
             for e in entries]
    return combine_from(cur, items, opt)


def cell_members(log: LogData, ana: Analysis, rpm_ax, map_ax, I: int, J: int):
    """Registros do log que caem na célula (I, J): (usados no cálculo, fora do filtro)."""
    rpm = log.col('RPM')
    mp = log.col('MAP')
    rlo = rpm_ax[max(J - 1, 0)]
    rhi = rpm_ax[min(J + 1, N - 1)]
    mlo = map_ax[max(I - 1, 0)]
    mhi = map_ax[min(I + 1, N - 1)]
    cr = (rpm > (-np.inf if J == 0 else rlo)) & (rpm < (np.inf if J == N - 1 else rhi))
    cm = (mp > (-np.inf if I == 0 else mlo)) & (mp < (np.inf if I == N - 1 else mhi))
    mem = cr & cm
    used = np.nonzero(mem & ana.ok)[0].tolist()
    other = np.nonzero(mem & ~ana.ok)[0].tolist()
    return used, other


BANDS = [(1500, 3000), (3000, 4000), (4000, 5000), (5000, 6000), (6000, 6500), (6500, 7600)]


def wot_summary(log: LogData, opt: dict) -> dict:
    """Carga plena (acelerador >= 80% por 12 registros) por faixa de rotação."""
    n, lag = log.n, opt['lag']
    cltMin = opt['cltC'] + 273
    L = log.col
    acc, clt, rpm = L('ACC %'), L('CLT'), L('RPM')
    l1, tgt, corr, util = L('Lambda 1'), L('Lambda Target'), L('Lambda Corr'), L('Inj. Utiliz.')
    buckets = [dict(a=a, b=b, lam=[], tgt=[], corr=[], util=[]) for a, b in BANDS]

    for t in range(11, n - lag):
        if not np.all(acc[t - 11:t + 1] >= 800):
            continue
        if not (clt[t] >= cltMin):
            continue
        r = rpm[t]
        bi = next((k for k, (a, b) in enumerate(BANDS) if a < r <= b), None)
        if bi is None:
            continue
        o = buckets[bi]
        o['lam'].append(l1[t + lag])
        o['tgt'].append(tgt[t])
        o['corr'].append(corr[t])
        if not math.isnan(util[t]):
            o['util'].append(util[t])

    def med(a):
        return float(np.median(a)) if a else math.nan

    rows = []
    for o in buckets:
        rows.append(dict(
            a=o['a'], b=o['b'], n=len(o['lam']),
            lam=med(o['lam']), tgt=med(o['tgt']), corr=med(o['corr']),
            util=float(max(o['util'])) if o['util'] else math.nan))

    knock = L('Knock')
    kmh = L('KM/H')
    return dict(
        rows=rows,
        knockNZ=int(np.sum(knock > 0)),
        knockCol=not math.isnan(knock[0]),
        rpmMax=float(np.nanmax(np.append(rpm, 0.0))),
        kmhMax=float(np.nanmax(np.append(kmh, 0.0))),
        utilMax=float(np.nanmax(np.append(util, 0.0))),
        n=n,
    )


# ---------- conveniência para o uso pelo arquivo ----------

def run(map_path: str, logs: list, prevs: list, opt: dict | None = None) -> dict:
    """logs: caminhos dos logs atuais (mesmo mapa). prevs: [(caminho_log, caminho_mapa), ...]
    na ordem do mais novo para o mais velho."""
    o = dict(DEFAULTS)
    o.update(opt or {})
    m = parse_map(load_text(map_path), map_path)
    entries = []
    for p in logs:
        entries.append(dict(log=parse_log(load_text(p)), base=m.fuel, weight=1.0))
    for k, (lp, mp_) in enumerate(prevs):
        base = parse_map(load_text(mp_), mp_).fuel
        entries.append(dict(log=parse_log(load_text(lp)), base=base, weight=0.5 ** (k + 1)))
    res = combine(m.fuel, m.rpm_ax, m.map_ax, entries, o)
    res['map'] = m
    res['entries'] = entries
    return res


def changed_cells(res: dict):
    m = res['map']
    out = []
    for i in range(N):
        for j in range(N):
            cur = m.fuel[i, j]
            nv = res['NEW'][i, j]
            if nv != cur:
                out.append(dict(i=i, j=j, kpa=m.map_ax[i], rpm=m.rpm_ax[j], cur=cur, new=nv,
                                pct=(nv / cur - 1) * 100, W=res['W'][i, j],
                                half=res['status'][i][j] == 'meia'))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Ajuste VE Peste Branca (Master Injection 2.0.35)')
    ap.add_argument('--map', required=True, help='mapa atual (.csv) que está no carro')
    ap.add_argument('--log', action='append', required=True, help='log gravado com o mapa atual (pode repetir)')
    ap.add_argument('--prev', nargs=2, action='append', default=[], metavar=('LOG', 'MAPA'),
                    help='log anterior e o mapa que estava no carro quando foi gravado (pode repetir; ordem do mais novo ao mais velho)')
    ap.add_argument('--saida', required=True, help='arquivo .csv de saída')
    ap.add_argument('--tol', type=float, default=1.0, help='tolerância em %% (padrão 1)')
    ap.add_argument('--aplicar', type=float, default=100.0, help='aplicar da correção em %% (padrão 100)')
    ap.add_argument('--max', type=float, default=15.0, help='mudança máx. por célula em %% (padrão 15)')
    ap.add_argument('--erro-grande', type=float, default=3.0, help='erro grande em %% (padrão 3)')
    ap.add_argument('--amostras-min', type=float, default=10, help='amostras mínimas (padrão 10)')
    ap.add_argument('--amostras-grande', type=float, default=4, help='amostras p/ erro grande (padrão 4)')
    ap.add_argument('--clt', type=float, default=82, help='motor quente acima de °C (padrão 82)')
    a = ap.parse_args(argv)

    opt = dict(
        tol=max(0.0, a.tol) / 100,
        bigTol=a.erro_grande / 100,
        minFull=a.amostras_min,
        minBig=a.amostras_grande,
        maxChange=a.max / 100,
        cltC=a.clt,
        strength=min(100.0, max(5.0, a.aplicar)) / 100,
    )
    res = run(a.map, a.log, a.prev, opt)
    m = res['map']
    with open(a.saida, 'w', encoding='latin-1', newline='') as f:
        f.write(write_map(m, res['NEW']))

    ch = changed_cells(res)
    print(f'{len(ch)} células alteradas -> {a.saida}')
    for c in ch:
        print(f"  MAP {c['kpa']:>5.0f} kPa  RPM {c['rpm']:>6.0f}  {c['cur']:>6.0f} -> {c['new']:>6.0f}"
              f"  ({c['pct']:+.1f}%)  amostras {c['W']:.1f}" + ('  [meia correção]' if c['half'] else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


# ---------- API para a interface (o front chama estas funções via servidor) ----------

def _clean(obj):
    """Converte NaN/inf em None e arrays numpy em listas, para JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def map_info(text: str) -> dict:
    m = parse_map(text)
    return _clean(dict(rpmAx=m.rpm_ax, mapAx=m.map_ax, fuel=m.fuel))


def log_info(text: str) -> dict:
    lg = parse_log(text)
    return dict(n=lg.n)


CHART_COLS = ['RPM', 'MAP', 'Lambda 1', 'Lambda Target', 'Lambda Corr', 'ACC %',
              'Ign. Adv.', 'Inj. Utiliz.', 'CLT', 'Knock', 'KM/H']


def calcular(p: dict) -> dict:
    """Payload: {map: texto, logs: [{text, base: texto|None}], chart: índice, opt: {...}}.
    Logs com base None são o log atual (peso 1, mapa atual). Os demais são anteriores,
    na ordem do mais novo para o mais velho, com pesos 0,5, 0,25...
    """
    o = dict(DEFAULTS)
    for k, v in (p.get('opt') or {}).items():
        if k in DEFAULTS:
            o[k] = float(v)
    o['lag'] = int(o['lag'])
    o['cltC'] = float(o['cltC'])

    m = parse_map(p['map'])
    logs = p.get('logs') or []
    if not logs:
        raise FormatError('Falta o log atual.')
    n_cur = sum(1 for e in logs if e.get('base') is None)
    if n_cur == 0:
        raise FormatError('Falta o log atual.')

    entries = []
    for idx, e in enumerate(logs):
        lg = parse_log(e['text'])
        if e.get('base') is None:
            entries.append(dict(log=lg, base=m.fuel, weight=1.0))
        else:
            base = parse_map(e['base']).fuel
            k = idx - n_cur
            entries.append(dict(log=lg, base=base, weight=0.5 ** (k + 1)))

    res = combine(m.fuel, m.rpm_ax, m.map_ax, entries, o)

    ci = min(max(int(p.get('chart', 0)), 0), len(entries) - 1)
    L = entries[ci]['log']
    ana = res['per'][ci]

    members = {}
    for I in range(N):
        for J in range(N):
            used, other = cell_members(L, ana, m.rpm_ax, m.map_ax, I, J)
            if used or other:
                members[f'{I},{J}'] = dict(used=used, other=other)

    chart = dict(
        n=L.n,
        cols={c: L.col(c) for c in CHART_COLS},
        ok=ana.ok,
        W=ana.W,
        C=ana.C,
        members=members,
        wot=wot_summary(L, o),
    )

    return _clean(dict(
        map=dict(rpmAx=m.rpm_ax, mapAx=m.map_ax, fuel=m.fuel),
        NEW=res['NEW'],
        mean=res['mean'],
        W=res['W'],
        status=res['status'],
        per=[dict(used=a.used) for a in res['per']],
        opt=o,
        chart=chart,
        csv=write_map(m, res['NEW']),
    ))
