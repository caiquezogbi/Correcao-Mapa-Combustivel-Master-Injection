"""Testes com dados sintéticos: mapa, log, regras e escrita do arquivo."""
import os
import tempfile

import numpy as np

import ajuste_ve as av


def make_map_text(fuel_value=800, eol='\r\n'):
    lines = ['Mapa de teste']
    rpm = [400 + k * 470 for k in range(16)]          # 400..7450
    mp = [10 + k * 10 for k in range(16)]              # 10..160
    for i in range(16):
        lines.append('#F%02d;' % (i + 1) + ';'.join([str(fuel_value)] * 16))
    for i in range(16):
        lines.append('#I%02d;' % (i + 1) + ';'.join(['20'] * 16))     # ignição (não mexe)
    lines.append('#I20;' + ';'.join(map(str, rpm)))
    lines.append('#I21;' + ';'.join(map(str, mp)))
    lines.append('#A01;' + ';'.join(['1000'] * 16))
    return eol.join(lines) + eol


def make_log_text(n=600, corr=1040, rpm_rng=(2000, 4000), map_rng=(60, 100), seed=1):
    rng = np.random.default_rng(seed)
    header = ['Mess 1', 'RPM', 'MAP', 'Lambda 1', 'Lambda Target', 'Lambda Corr',
              'Lambda Loop', 'CLT', 'ACC %', 'Inj. Utiliz.', 'Knock', 'IAT', 'KM/H', 'Ign. Adv.']
    rows = ['Mess 1;' + ';'.join(header[1:])]
    rpm = mp = 0
    for k in range(n):
        if k % 20 == 0:                      # cada ponto de operação dura 20 registros (log real é estável assim)
            rpm = rng.integers(*rpm_rng)
            mp = rng.integers(*map_rng)
        vals = [rpm, mp, 1000, 1000, corr, 1, 360, 0, 300, 0, 300, 0, 10]
        rows.append('#D01;' + ';'.join(str(v) for v in vals))
    return '\r\n'.join(rows) + '\r\n'


def test_parse_and_write_keep_other_lines():
    text = make_map_text()
    m = av.parse_map(text)
    assert m.fuel.shape == (16, 16)
    assert np.all(m.fuel == 800)
    new = m.fuel.copy()
    new[3, 4] = 831
    out = av.write_map(m, new)
    assert out.count('\r\n') == text.count('\r\n')
    assert '#F04;800;800;800;800;831;' in out
    assert '#I01;20;20' in out and '#A01;1000' in out     # outras linhas intactas
    assert av.parse_map(out).fuel[3, 4] == 831


def test_bad_map_raises():
    try:
        av.parse_map('lixo\r\n')
    except av.FormatError as e:
        assert '#F01' in str(e)
    else:
        raise AssertionError('deveria falhar')


def test_log_parse_and_window():
    log = av.parse_log(make_log_text(n=120))
    assert log.n == 120
    mx, mn = av.fwd_max_min(np.array([1.0, 5.0, 3.0, np.nan, 2.0]), 2)
    assert mx[0] == 5.0 and mn[0] == 1.0
    assert mx[3] == 2.0 and mn[3] == 2.0
    assert mx[4] == 2.0


def test_correction_applied_in_populated_cells():
    # Lambda Corr 1040 em todo o log -> erro +4% onde há amostras suficientes
    m_text = make_map_text(800)
    log_text = make_log_text(n=600, corr=1040)
    with tempfile.TemporaryDirectory() as d:
        mp = os.path.join(d, 'CLAUDE 6.csv')
        lp = os.path.join(d, 'log.csv')
        open(mp, 'w', encoding='latin-1', newline='').write(m_text)
        open(lp, 'w', encoding='latin-1', newline='').write(log_text)
        res = av.run(mp, [lp], [])
    ch = av.changed_cells(res)
    assert len(ch) > 0
    # células dentro da faixa de dados sobem de 800 para 832 (+4%)
    for c in ch:
        # 832 = correção inteira (+4%); 816 = meia correção (+2%) em célula com 1,5 a 4 amostras
        if c['W'] >= 4:
            assert c['new'] == 832 and not c['half'], c
        else:
            assert c['new'] == 816 and c['half'], c
        # bilinear: células vizinhas da faixa de dados também recebem parte da amostra
        assert 1000 <= c['rpm'] <= 5000 and 30 <= c['kpa'] <= 130, c
    # célula fora dos dados não muda
    assert res['NEW'][15, 15] == 800


def test_weight_and_previous_map_used():
    # log anterior gravado com mapa 1000: erro medido vs mapa atual 800 é 0 (ECU sem correção)
    m_text = make_map_text(800)
    log_now = make_log_text(n=600, corr=1040, seed=2)
    log_prev = make_log_text(n=600, corr=1000, seed=3)
    with tempfile.TemporaryDirectory() as d:
        mp = os.path.join(d, 'atual.csv')
        mpp = os.path.join(d, 'anterior.csv')
        open(mp, 'w', encoding='latin-1', newline='').write(m_text)
        open(mpp, 'w', encoding='latin-1', newline='').write(make_map_text(1000))
        lp = os.path.join(d, 'log.csv')
        lpp = os.path.join(d, 'log_ant.csv')
        open(lp, 'w', encoding='latin-1', newline='').write(log_now)
        open(lpp, 'w', encoding='latin-1', newline='').write(log_prev)
        res = av.run(mp, [lp], [(lpp, mpp)])
    assert res['entries'][1]['base'][0, 0] == 1000
    assert abs(res['entries'][1]['weight'] - 0.5) < 1e-12


def test_round_half_even_and_limits():
    a = av.Analysis(W=np.zeros((16, 16)), S=np.zeros((16, 16)), C=np.zeros((16, 16)),
                    used=0, ok=np.zeros(1, bool), f=np.zeros(1))
    opt = dict(av.DEFAULTS)
    cur = np.full((16, 16), 1000.0)
    W = np.zeros((16, 16)); S = np.zeros((16, 16))
    W[0, 0] = 20; S[0, 0] = 20 * 1.5               # erro +50%, limitado a +15%
    a.W, a.S = W, S
    r = av.combine_from(cur, [dict(a=a, base=cur, weight=1.0)], opt)
    assert r['NEW'][0, 0] == 1150


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
