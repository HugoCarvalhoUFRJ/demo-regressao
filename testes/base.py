"""Infraestrutura comum dos testes no navegador: servidor HTTP local, Chrome com perfil temporário e utilidades."""
import http.server
import threading
from functools import partial

import numpy as np

from comum import PROJ

PORTA = 8766
BASE = f'http://127.0.0.1:{PORTA}'
X0, Y1, ML, MT, S = -1.73, 3.75, 54, 8, 100


class Quieto(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def servidor():
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', PORTA), partial(Quieto, directory=str(PROJ)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def novo_navegador(p):
    return p.chromium.launch(channel='chrome', headless=True)


def coleta_erros(page, lista):
    def console(m):
        if m.type == 'error' and 'fonts.g' not in m.text and 'ERR_' not in m.text and 'favicon' not in (m.location or {}).get('url', ''):
            lista.append(m.text)
    page.on('console', console)
    page.on('pageerror', lambda e: lista.append(str(e)))


def espera_quadro(page):
    page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
    page.wait_for_timeout(30)


def arrasta(page, de, para, passos=5):
    box = page.locator('#plot').bounding_box()
    f = box['width'] / 602
    a = (box['x'] + de[0] * f, box['y'] + de[1] * f)
    b = (box['x'] + para[0] * f, box['y'] + para[1] * f)
    page.mouse.move(*a)
    page.mouse.down()
    for t in np.linspace(0, 1, passos + 1)[1:]:
        page.mouse.move(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        espera_quadro(page)
    page.mouse.up()
    espera_quadro(page)


def tela(x, y):
    """Coordenadas de dados para coordenadas do viewBox do SVG."""
    return ML + (x - X0) * S, MT + (Y1 - y) * S
