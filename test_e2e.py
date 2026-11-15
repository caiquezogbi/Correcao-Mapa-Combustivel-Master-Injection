"""Teste de ponta a ponta: servidor Python + página no Chrome (headless).

Confere que a página calcula, que o número de células bate com o cálculo direto
do núcleo, que o log anterior exige mapa, e que o download traz só as linhas #F.
"""
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright

import ajuste_ve as av
import test_ajuste_ve as t

PORT = 8799
URL = f'http://127.0.0.1:{PORT}/'
CHROME = '/usr/bin/google-chrome'


def wait_server():
    for _ in range(100):
        try:
            urllib.request.urlopen(URL, timeout=1).read()
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError('servidor não subiu')


def write(path, text):
    with open(path, 'w', encoding='latin-1', newline='') as f:
        f.write(text)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    d = tempfile.mkdtemp()
    map_atual = os.path.join(d, 'CLAUDE 6.csv')
    log_atual = os.path.join(d, 'log_atual.csv')
    map_ant = os.path.join(d, 'CORRECAO 5.csv')
    log_ant = os.path.join(d, 'log_ant.csv')
    write(map_atual, t.make_map_text(800))
    write(log_atual, t.make_log_text(600, corr=1040))
    write(map_ant, t.make_map_text(1000))
    write(log_ant, t.make_log_text(600, corr=1000, seed=3))
    bad_map = os.path.join(d, 'ruim.csv')
    write(bad_map, 'isto nao e um mapa\r\n')

    # resultado de referência: cálculo direto no núcleo (mesmos arquivos, sem prévio)
    ref = av.run(map_atual, [log_atual], [])
    ref_n = len(av.changed_cells(ref))

    srv = subprocess.Popen([sys.executable, os.path.join(here, 'server.py'), '--port', str(PORT)],
                           cwd=here, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    errors = []
    try:
        wait_server()
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=CHROME, headless=True, args=['--no-sandbox'])
            page = browser.new_page(viewport={'width': 1280, 'height': 900})
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
            page.goto(URL)
            page.wait_for_selector('#btn-run')

            # 1) mapa ruim: mensagem de erro, nada calcula
            page.set_input_files('#in-map', bad_map)
            page.wait_for_selector('#map-err:not([hidden])')
            assert 'mapa do Master Injection' in page.inner_text('#map-err'), page.inner_text('#map-err')

            # 2) mapa certo + log atual
            page.set_input_files('#in-map', map_atual)
            page.wait_for_selector('#map-info:not([hidden])')
            assert page.input_value('#out-name') == 'CLAUDE 7.csv', page.input_value('#out-name')
            page.set_input_files('#in-log', log_atual)
            page.wait_for_selector('#log-list .file')
            page.wait_for_function("document.querySelector('#btn-run') && !document.querySelector('#btn-run').disabled")
            page.click('#btn-run')
            page.wait_for_selector('#results:not([hidden])', timeout=15000)
            chips = page.inner_text('#chips')
            m = re.search(r'(\d+)\s*\n?\s*células alteradas', chips.replace('\n', ' '))
            n_page = int(m.group(1)) if m else -1
            assert n_page == ref_n, f'página {n_page} x núcleo {ref_n}: {chips!r}'
            print('ok: células alteradas na página =', n_page, '(igual ao núcleo)')

            # 3) log anterior sem mapa: botão Calcular desligado e aviso
            page.set_input_files('#in-prev-log', log_ant)
            page.wait_for_selector('#prev-list .file.need')
            assert page.is_disabled('#btn-run'), 'deveria bloquear sem mapa do log anterior'
            assert 'Falta escolher o mapa' in page.inner_text('#run-status')

            # 4) escolhe o mapa do log anterior (o seletor de arquivo é simulado)
            page.on('filechooser', lambda fc: fc.set_files(map_ant))
            page.click('#prev-map-0')
            page.wait_for_selector('#prev-list .file:not(.need)')
            page.wait_for_function("!document.querySelector('#btn-run').disabled")
            page.click('#btn-run')
            page.wait_for_function("document.querySelector('#run-status').textContent !== 'Calculando…'")
            assert 'Erro' not in page.inner_text('#run-status'), page.inner_text('#run-status')
            page.wait_for_selector('#results:not([hidden])')
            print('ok: com log anterior e mapa escolhido, calcula sem erro')

            # 5) clicar numa célula alterada abre a info da célula
            page.click('#chg tbody tr:first-child')
            page.wait_for_selector('#cellinfo:not([hidden])')
            assert 'erro medido' in page.inner_text('#cellinfo')
            print('ok: clique na lista abre a célula e o gráfico')

            # 6) download: só as linhas #F mudam
            with page.expect_download() as dl_info:
                page.click('#btn-save')
            dl = dl_info.value
            out_path = os.path.join(d, dl.suggested_filename)
            dl.save_as(out_path)
            with open(out_path, encoding='latin-1', newline='') as f:
                saida = f.read()
            orig = t.make_map_text(800)
            orig_lines = orig.split('\r\n')
            out_lines = saida.split('\r\n')
            assert len(orig_lines) == len(out_lines), 'número de linhas mudou'
            diff = [i for i, (a, b) in enumerate(zip(orig_lines, out_lines)) if a != b]
            assert diff and all(out_lines[i].startswith('#F') for i in diff), 'mudou linha que não é #F'
            assert saida.count('\r\n') == orig.count('\r\n')
            print('ok: download', dl.suggested_filename, '- linhas alteradas:', len(diff), '(todas #F)')

            page.click('#btn-run') if not page.is_disabled('#btn-run') else None
            browser.close()
    finally:
        srv.terminate()
        srv.wait(timeout=5)

    # 'Failed to load resource' = 400 de propósito (mapa inválido), favicon e fontes offline.
    # Erros de JavaScript (pageerror) não entram nessa lista e reprovam o teste.
    real_errors = [e for e in errors if not e.startswith('Failed to load resource')]
    assert not real_errors, real_errors
    print('ok: sem erros no console')


if __name__ == '__main__':
    main()
