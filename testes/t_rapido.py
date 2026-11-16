"""Verificação rápida durante o desenvolvimento: abre um app, exercita os controles, junta erros e tira capturas.
Uso: python t_rapido.py <app> [<app> ...]   (capturas em resultados-regressao/capturas/rapido_<app>_<nome>.png)
"""
import re
import sys

from playwright.sync_api import sync_playwright

from base import BASE, arrasta, coleta_erros, espera_quadro, novo_navegador, servidor, tela
from comum import RES

ACOES = {
    'comum': [
        ("shot", 'inicio'),
        ("check", '#o-test'), ("check", '#o-truef'), ("hover", tela(1.0, 0)), ("shot", 'teste'),
        ("check", '#o-var'), ("wait", 500), ("hover", tela(1.5, 0)), ("shot", 'var'),
        ("select", '#dataset', 'degrau'), ("shot", 'degrau'),
        ("select", '#dataset', 'influente'), ("shot", 'influente'),
        ("click", '#vseg button[data-mode="2"]'), ("shot", 'dois'),
        ("hover", tela(1.0, 1.0)), ("shot", 'dois-sonda'),
        ("select", '#dataset', 'degrau2'), ("shot", 'dois-degrau'),
        ("dark", None), ("shot", 'dois-escuro'),
        ("click", '#vseg button[data-mode="1"]'), ("shot", 'um-escuro'), ("light", None),
    ],
    'knn': [("click", '#wseg button[data-w="distance"]'), ("shot", 'pesos'), ("click", '#wseg button[data-w="uniform"]'),
            ("click", '#vseg button[data-mode="2"]'), ("click", '#pseg button[data-p="1"]'), ("hover", tela(0.5, 0.5)), ("shot", 'p1'),
            ("click", '#vseg button[data-mode="1"]')],
    'penalizada': [("click", '#vseg button[data-mode="2"]'), ("select", '#dataset', 'corr'), ("shot", 'p-inicio'), ("fill", '#p-lam-n', '40'), ("shot", 'p-lasso40'),
                   ("click", '#penseg button[data-pen="ridge"]'), ("shot", 'p-ridge'),
                   ("click", '#penseg button[data-pen="enet"]'), ("shot", 'p-enet'),
                   ("click", '#vseg button[data-mode="1"]'), ("click", '#penseg button[data-pen="lasso"]'), ("shot", 'p-um'),
                   ("click", '#vseg button[data-mode="2"]')],
    'arvore': [("hovernode", 2), ("shot", 'a-no2'), ("fill", '#p-alpha-n', '1.5'), ("shot", 'a-poda'),
               ("click", '#vseg button[data-mode="2"]'), ("hover", tela(2.5, 2.5)), ("shot", 'a-dois'), ("hovernode", 1), ("shot", 'a-dois-no1')],
    'floresta': [("check", '#o-test'), ("shot", 'f-um'), ("click", '#baseseg button[data-base="linear"]'), ("shot", 'f-linear'), ("click", '#baseseg button[data-base="tree"]'),
                 ("click", '#vseg button[data-mode="2"]'), ("shot", 'f-dois'), ("fill", '#p-b-n', '500'), ("shot", 'f-500'), ("click", '#vseg button[data-mode="1"]')],
    'boosting': [("fill", '#p-st-n', '5'), ("shot", 'b-5'), ("fill", '#p-st-n', '60'), ("shot", 'b-60'), ("fill", '#p-d-n', '3'), ("click", '#vseg button[data-mode="2"]'), ("shot", 'b-dois'), ("click", '#play'), ("wait", 600), ("shot", 'b-anim'), ("click", '#vseg button[data-mode="1"]')],
    'linear': [("select", '#dataset', 'reta'), ("check", '#o-sq'), ("hoverel", '#v-rss', (0.3, 0.3)), ("shot", 'rss'),
               ("select", '#dataset', 'pouco'), ("fill", '#p-d-n', '9'), ("shot", 'grau9'), ("fill", '#p-d-n', '10'), ("shot", 'grau10'),
               ("select", '#dataset', 'curva'), ("fill", '#p-d-n', '4'), ("shot", 'curva4'),
               ("click", '#vseg button[data-mode="2"]'), ("check", '#o-inter'), ("select", '#dataset', 'inter'), ("shot", 'inter'),
               ("click", '#vseg button[data-mode="1"]')],
}


def confere_knn(pg):
    """Com uma variável, a vizinhança desenhada (linhas tracejadas) tem a mesma largura com p = 1, 2 e 3."""
    pg.click('#vseg button[data-mode="1"]')
    espera_quadro(pg)
    box = pg.locator('#plot').bounding_box()
    pg.mouse.move(box['x'] + 300 * box['width'] / 602, box['y'] + 300 * box['width'] / 602)
    larg = {}
    for p in (1, 2, 3):
        pg.click(f'#pseg button[data-p="{p}"]')
        espera_quadro(pg)
        pg.mouse.move(box['x'] + 301 * box['width'] / 602, box['y'] + 300 * box['width'] / 602)
        espera_quadro(pg)
        d = pg.get_attribute('#probe-p', 'd') or ''
        xs = [float(v) for v in re.findall(r'M([-\d.]+)', d)]
        larg[p] = round(max(xs) - min(xs), 3) if len(xs) >= 2 else None
    ok = larg[1] is not None and larg[1] == larg[2] == larg[3]
    return [f'largura da vizinhança (px do viewBox) com p = 1, 2, 3: {larg} → {"ok" if ok else "PROBLEMA"}']


def confere_penalizada(pg):
    """Estado inicial: duas variáveis, lasso, um coeficiente exatamente zero (mínimo num vértice do losango)."""
    pg.goto(pg.url)
    espera_quadro(pg)
    r = pg.evaluate("() => { const m = window.__regr.state.model; return { modo: window.__regr.state.mode, pen: m.pen, lam: m.lam, b: Array.from(m.b) }; }")
    zeros = sum(v == 0 for v in r['b'])
    return [f'início: modo {r["modo"]}, {r["pen"]}, λ = {r["lam"]}, β = {r["b"]} → {zeros} zero(s): {"ok" if r["modo"] == 2 and r["pen"] == "lasso" and zeros == 1 else "PROBLEMA"}']


CONFERE = {'knn': confere_knn, 'penalizada': confere_penalizada}


def main(apps, mobile=False):
    srv = servidor()
    with sync_playwright() as p:
        b = novo_navegador(p)
        for app in apps:
            vp = {'width': 390, 'height': 844} if mobile else {'width': 1300, 'height': 900}
            pg = b.new_page(viewport=vp, has_touch=mobile, is_mobile=mobile)
            erros = []
            coleta_erros(pg, erros)
            pg.goto(f'{BASE}/{app}.html')
            espera_quadro(pg)
            for a in ACOES['comum'] + ACOES.get(app, []):
                kind, alvo = a[0], a[1]
                if kind == 'select':
                    if not pg.evaluate(f"() => !!document.querySelector('{alvo} option[value=\"{a[2]}\"]')"):
                        continue
                    pg.select_option(alvo, a[2])
                elif kind == 'check':
                    if pg.locator(alvo).is_visible():
                        pg.check(alvo)
                elif kind == 'click':
                    if pg.locator(alvo).count() and pg.locator(alvo).is_visible():
                        pg.click(alvo)
                elif kind == 'fill':
                    pg.fill(alvo, a[2])
                elif kind == 'hover':
                    if mobile:
                        continue
                    box = pg.locator('#plot').bounding_box()
                    pg.mouse.move(box['x'] + alvo[0] * box['width'] / 602, box['y'] + alvo[1] * box['width'] / 602)
                elif kind == 'hoverel':
                    box = pg.locator(alvo).bounding_box()
                    pg.mouse.move(box['x'] + a[2][0] * box['width'], box['y'] + a[2][1] * box['height'])
                elif kind == 'hovernode':
                    pg.locator('#v-tree .tn').nth(alvo).hover()
                elif kind == 'wait':
                    pg.wait_for_timeout(alvo)
                elif kind == 'drag':
                    arrasta(pg, alvo, a[2])
                elif kind in ('dark', 'light'):
                    pg.evaluate(f"() => document.documentElement.setAttribute('data-theme', '{kind}')")
                espera_quadro(pg)
                if kind == 'shot':
                    sufixo = '_cel' if mobile else ''
                    pg.screenshot(path=str(RES / 'capturas' / f'rapido_{app}_{alvo}{sufixo}.png'), full_page=True)
            txt = pg.evaluate("() => document.getElementById('v-data').innerText.replace(/\\s+/g, ' ') + ' | ' + document.getElementById('say').innerText")
            sw = pg.evaluate("() => document.documentElement.scrollWidth")
            print(app, 'erros:', erros, '| scrollWidth', sw, '|', txt)
            for linha in CONFERE.get(app, lambda pg: [])(pg):
                print('  ', linha)
            pg.close()
        b.close()
    srv.shutdown()


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--cel']
    main(args or ['knn'], mobile='--cel' in sys.argv)
