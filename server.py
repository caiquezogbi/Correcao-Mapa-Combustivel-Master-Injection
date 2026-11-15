"""Servidor do Ajuste VE: entrega a página (HTML/CSS/JS) e faz o cálculo em Python.

Só biblioteca padrão + numpy. Sem banco de dados e sem estado entre chamadas:
o navegador manda os textos dos arquivos e recebe o resultado.

Uso:
    python server.py                 # só este computador: http://127.0.0.1:8765
    python server.py --host 0.0.0.0  # outros aparelhos da rede (ver aviso abaixo)

Aviso: não há login. Para deixar exposto na internet, coloque atrás de um
proxy com autenticação (por exemplo, Nginx com senha ou VPN).
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ajuste_ve as av

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, 'web', 'index.html')
MAX_BODY = 200 * 1024 * 1024   # 200 MB: logs grandes em texto

ROUTES = {
    '/api/mapa': lambda p: av.map_info(p['text']),
    '/api/log': lambda p: av.log_info(p['text']),
    '/api/calcular': lambda p: av.calcular(p),
}


class Handler(BaseHTTPRequestHandler):
    server_version = 'AjusteVE/1.0'

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            with open(PAGE, 'rb') as f:
                self._send(200, 'text/html; charset=utf-8', f.read())
        else:
            self._send(404, 'text/plain; charset=utf-8', b'nao encontrado')

    def do_POST(self):
        fn = ROUTES.get(self.path)
        if fn is None:
            self._send(404, 'text/plain; charset=utf-8', b'nao encontrado')
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY:
            self._json(413 if length > MAX_BODY else 400, {'error': 'Pedido vazio ou grande demais.'})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode('utf-8'))
            self._json(200, fn(payload))
        except av.FormatError as e:
            self._json(400, {'error': str(e)})
        except Exception as e:  # erro inesperado: mensagem sem trace para a tela
            self._json(500, {'error': 'Erro interno no cálculo: ' + str(e)})

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self._send(code, 'application/json; charset=utf-8', body)

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # silencioso
        pass


def make_server(host='127.0.0.1', port=8765):
    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    ap = argparse.ArgumentParser(description='Ajuste VE Peste Branca - servidor')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765)
    a = ap.parse_args(argv)
    srv = make_server(a.host, a.port)
    print(f'Ajuste VE rodando em http://{a.host}:{a.port}  (Ctrl+C para parar)')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
