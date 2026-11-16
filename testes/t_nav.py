"""Testes no navegador (Chrome já instalado, perfil temporário): ponta a ponta, interação aleatória, acessibilidade, contextos, desempenho e capturas.
Uso: python t_nav.py [e2e|aleatorio|axe|contextos|desempenho|capturas|todos]
"""
import os
import random
import re
import subprocess
import sys
import tempfile
import traceback
import warnings

import numpy as np
from axe_playwright_python.sync_playwright import Axe
from playwright.sync_api import sync_playwright

from base import BASE, arrasta, coleta_erros, espera_quadro, novo_navegador, servidor, tela
from comum import PROJ, RES, RS, relatorio

warnings.filterwarnings('ignore')
APPS = ['linear', 'penalizada', 'knn', 'arvore', 'floresta', 'boosting']

# ------------------------------------------------------------------ leitura do estado e execução literal do código mostrado
LE = """() => { const r = window.__regr, s = r.state, m = s.model;
  const pts = s.points.map(p => s.mode === 1 ? [p.x, p.y] : [p.x, p.y, p.v]);
  const tst = s.test.map(p => s.mode === 1 ? [p.x, p.y] : [p.x, p.y, p.v]);
  return { mode: s.mode, pts, tst, ok: m.ok, warn: m.warn || '', P: s.P, py: document.getElementById('code-py').textContent, r: document.getElementById('code-r').textContent,
           dados: document.getElementById('v-data').innerText, say: document.getElementById('say').innerText }; }"""
PREV = """(G) => { const m = window.__regr.state.model; return G.map(g => m.f(g[0], g[1])); }"""


def grade(mode):
    if mode == 1:
        return np.column_stack([np.linspace(-1.6, 3.4, 120), np.zeros(120)])
    a, b = np.meshgrid(np.linspace(-1.6, 3.4, 23), np.linspace(-1.65, 3.6, 23))
    return np.column_stack([a.ravel(), b.ravel()])


def xy(e):
    P = np.array(e['pts'])
    return (P[:, :1], P[:, 1]) if e['mode'] == 1 else (P[:, :2], P[:, 2])


def py_executa(codigo, X, y, G):
    """Executa literalmente as linhas de Python mostradas; devolve as previsões na grade (ou o valor da última linha)."""
    from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor  # noqa: F401
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge  # noqa: F401
    from sklearn.neighbors import KNeighborsRegressor  # noqa: F401
    from sklearn.pipeline import make_pipeline  # noqa: F401
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: F401
    from sklearn.tree import DecisionTreeRegressor  # noqa: F401
    import statsmodels.api as sm  # noqa: F401
    ns = dict(locals())
    ns.update(X=X, y=y, X_novos=G)
    linhas = [l for l in codigo.split('\n') if l.strip() and not l.strip().startswith('#')]
    for l in linhas[:-1]:
        exec(l, ns)
    ult = linhas[-1].split('#')[0].strip()
    try:
        val = eval(ult, ns)
    except SyntaxError:
        exec(ult, ns); val = None
    if 'staged_predict' in ult:
        return np.asarray(val)
    return ns['modelo'].predict(G)


def r_executa(codigo, X, y, G):
    """Executa literalmente as linhas de R mostradas; devolve o valor da última expressão, ou as previsões em x0."""
    with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as d:
        np.savetxt(os.path.join(d, 'P.csv'), np.column_stack([X, y]), delimiter=',')
        np.savetxt(os.path.join(d, 'G.csv'), G, delimiter=',')
        open(os.path.join(d, 'linha.txt'), 'w', encoding='utf-8').write(codigo)
        scr = r"""
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({library(rpart); library(randomForest); library(FNN); library(glmnet); library(gbm); suppressWarnings(library(kknn))})
d <- commandArgs(TRUE)[1]; A <- as.matrix(read.csv(file.path(d, "P.csv"), header = FALSE)); G <- as.matrix(read.csv(file.path(d, "G.csv"), header = FALSE))
p <- ncol(A) - 1
if (p == 1) { dados <- data.frame(x = A[, 1], y = A[, 2]); x0 <- data.frame(x = G[, 1]) } else { dados <- data.frame(X1 = A[, 1], X2 = A[, 2], y = A[, 3]); x0 <- data.frame(X1 = G[, 1], X2 = G[, 2]) }
X <- dados[, setdiff(names(dados), "y"), drop = FALSE]; y <- dados$y; novos <- x0; x <- if (p == 1) dados$x else NULL
if (p == 2) X <- as.matrix(X)
set.seed(1)
exprs <- parse(text = readLines(file.path(d, "linha.txt"), encoding = "UTF-8"))
for (e in exprs) r <- suppressWarnings(eval(e, envir = globalenv()))
out <- if (exists("previsto")) previsto else if (inherits(r, "summary.lm")) predict(modelo, x0) else if (inherits(r, "dgCMatrix") || is.matrix(r)) as.numeric(r) else if (inherits(r, "rpart") || inherits(r, "randomForest")) predict(r, x0) else as.numeric(r)
if (exists("modelo") && (inherits(modelo, "rpart") || inherits(modelo, "randomForest")) && !exists("previsto")) out <- predict(modelo, x0)
cat(sprintf("%.17g", out), sep = "\n")
"""
        f = os.path.join(d, 'r.R'); open(f, 'w', encoding='utf-8').write(scr)
        r = subprocess.run([RS, f, d], capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode:
            raise RuntimeError(r.stderr[-600:])
        return np.array([float(v) for v in r.stdout.split()])


def numeros_pt(txt):
    return [float(v.replace('.', '').replace(',', '.').replace('−', '-')) for v in re.findall(r'−?\d[\d.]*,\d+', txt)]


# ------------------------------------------------------------------ estados ponta a ponta
# (modo, conjunto, ações) — ações: ('fill', id, valor), ('click', seletor), ('check', id)
ESTADOS = {
    'linear': [(1, 'reta', []), (1, 'curva', [('fill', '#p-d-n', '4')]), (1, 'pouco', [('fill', '#p-d-n', '9')]), (1, 'influente', []),
               (2, 'plano', []), (2, 'inter', [('check', '#o-inter')])],
    'penalizada': [(2, 'corr', []), (2, 'corr', [('click', '#penseg button[data-pen="ridge"]'), ('fill', '#p-lam-n', '50')]),
                   (2, 'plano', [('click', '#penseg button[data-pen="enet"]'), ('fill', '#p-alpha-n', '0.3'), ('fill', '#p-lam-n', '25')]),
                   (2, 'so1', [('click', '#o-std'), ('fill', '#p-lam-n', '40')]), (1, 'reta', [('fill', '#p-lam-n', '30')]),
                   (1, 'curva', [('click', '#penseg button[data-pen="ridge"]'), ('fill', '#p-lam-n', '100')]),
                   # Polinômio (uma variável): colunas x, …, x^d padronizadas; lasso, ridge e elastic net
                   (1, 'curva', [('fill', '#p-d-n', '4'), ('fill', '#p-lam-n', '2')]),
                   (1, 'seno', [('click', '#penseg button[data-pen="ridge"]'), ('fill', '#p-d-n', '6'), ('fill', '#p-lam-n', '0.5')]),
                   (1, 'quad', [('click', '#penseg button[data-pen="enet"]'), ('fill', '#p-alpha-n', '0.5'), ('fill', '#p-d-n', '3'), ('fill', '#p-lam-n', '5')])],
    'knn': [(1, 'curva', []), (1, 'seno', [('fill', '#p-k-n', '1')]), (1, 'degrau', [('click', '#wseg button[data-w="distance"]'), ('fill', '#p-k-n', '7')]),
            (2, 'chapeu', []), (2, 'sela', [('click', '#pseg button[data-p="1"]'), ('fill', '#p-k-n', '9')]),
            (2, 'plano', [('click', '#pseg button[data-p="3"]'), ('click', '#wseg button[data-w="distance"]')])],
    'arvore': [(1, 'degrau', []), (1, 'seno', [('fill', '#p-depth-n', '3')]), (1, 'curva', [('fill', '#p-alpha-n', '0.8')]),
               (2, 'degrau2', []), (2, 'chapeu', [('fill', '#p-leaf-n', '4')]), (2, 'sela', [('fill', '#p-alpha-n', '0.3')])],
    'floresta': [(1, 'seno', []), (1, 'curva', [('click', '#baseseg button[data-base="linear"]')]), (2, 'chapeu', []),
                 (2, 'degrau2', [('click', '#mseg button[data-m="2"]'), ('fill', '#p-leaf-n', '3')]), (2, 'plano', [('click', '#baseseg button[data-base="linear"]')])],
    'boosting': [(1, 'seno', []), (1, 'degrau', [('fill', '#p-d-n', '3'), ('fill', '#p-lr-n', '0.2'), ('fill', '#p-st-n', '40')]),
                 (2, 'chapeu', [('fill', '#p-nt-n', '200')]), (2, 'degrau2', [('fill', '#p-d-n', '2'), ('fill', '#p-st-n', '25')])],
}


def prepara_estado(pg, app, modo, conj, acoes):
    pg.goto(f'{BASE}/{app}.html')
    espera_quadro(pg)
    pg.click(f'#vseg button[data-mode="{modo}"]')
    espera_quadro(pg)
    pg.select_option('#dataset', conj)
    espera_quadro(pg)
    if pg.locator('#o-test').is_enabled(): pg.check('#o-test')
    for a in acoes:
        if a[0] == 'fill': pg.fill(a[1], a[2])
        elif a[0] == 'click': pg.click(a[1])
        elif a[0] == 'check': pg.check(a[1])
        espera_quadro(pg)
    # Arrasta o primeiro ponto (na posição da tela) para outro lugar
    p0 = pg.evaluate("() => { const p = window.__regr.state.points[0]; return [p.x, p.y]; }")
    arrasta(pg, tela(*p0), tela(p0[0] + 0.4, p0[1] - 0.5 if modo == 1 else p0[1] + 0.3))
    pg.mouse.move(2, 2)
    espera_quadro(pg)


def e2e():
    linhas, falhas = [], 0
    srv = servidor()
    with sync_playwright() as p:
        b = novo_navegador(p)
        for app in APPS:
            for modo, conj, acoes in ESTADOS[app]:
                pg = b.new_page(viewport={'width': 1300, 'height': 900})
                erros = []
                coleta_erros(pg, erros)
                nome = f'{app} {modo}v {conj} {acoes}'
                try:
                    prepara_estado(pg, app, modo, conj, acoes)
                    e = pg.evaluate(LE)
                    X, y = xy(e)
                    G = grade(modo)
                    app_g = np.array(pg.evaluate(PREV, G.tolist()))
                    res = []
                    # Python literal
                    ref = py_executa(e['py'], X, y, G[:, :X.shape[1]])
                    dif = float(np.max(np.abs(app_g - ref))) if app != 'floresta' else float(np.mean(np.abs(app_g - ref)))
                    estocastico = app == 'floresta' and 'Bagging' not in e['py'] and 'LinearRegression' not in e['py']
                    if app == 'floresta':
                        tol = 0.15 * (np.std(y) + 1e-9) if estocastico else 0.08 * (np.std(y) + 1e-9)
                    else:
                        tol = 1e-6
                    ok_py = dif <= tol
                    empate = ''
                    if not ok_py and app in ('arvore', 'boosting'):
                        # Empate de ganho entre X₁ e X₂ (o scikit-learn sorteia a ordem dos atributos): mesmo ajuste nos pontos de treino?
                        app_t = np.array(pg.evaluate(PREV, np.column_stack([X, np.zeros(len(X))]).tolist() if modo == 1 else X.tolist()))
                        if float(np.max(np.abs(app_t - py_executa(e['py'], X, y, X)))) <= 1e-9:
                            ok_py, empate = True, '; diferença só entre os pontos: empate de ganho entre X₁ e X₂, mesmo ajuste nos pontos de treino'
                    res.append(f'Python {"ok" if ok_py else "DIVERGE"} ({"diferença média" if app == "floresta" else "maior diferença"} {dif:.2e}{", outra semente: só a ordem de grandeza" if app == "floresta" else ""}{empate})')
                    # MSE de treino e de teste mostrados contra os calculados com o modelo do Python (onde é determinístico)
                    if app not in ('floresta',) and not (app == 'boosting'):
                        nums = numeros_pt(e['dados'])
                        mse_tr = float(np.mean((y - py_executa(e['py'], X, y, X)) ** 2))
                        ok_mse = abs(nums[0] - mse_tr) <= 1.5e-3
                        res.append(f'MSE de treino na tela {nums[0]:.3f} contra {mse_tr:.4f}: {"ok" if ok_mse else "DIVERGE"}')
                        ok_py = ok_py and ok_mse
                    # R literal
                    try:
                        rv = r_executa(e['r'], X, y, G[:, :X.shape[1]])
                        if app == 'penalizada':
                            m = pg.evaluate("() => { const m = window.__regr.state.model; return { b: Array.from(m.b), b0: m.b0, raw: m.raw, my: m.pr.my, std: m.std, one: m.pr.p === 1 }; }")
                            # O que a última linha de R devolve: com uma variável, (b0, inclinação) ou só b0; com duas, coef() na escala de Xs
                            if m['one']: alvo = np.r_[m['b0'], m['raw']] if m['std'] else np.r_[m['b0']]
                            else: alvo = np.r_[m['my'] if m['std'] else m['b0'], m['b']]
                            dr = float(np.max(np.abs(rv - alvo)))
                            okr = dr <= 1e-4
                        elif app == 'boosting' or app == 'floresta':
                            dr = float(np.mean(np.abs(rv - app_g))) if len(rv) == len(app_g) else float(np.max(np.abs(rv[:len(X[0]) + 1])))
                            okr = True
                        else:
                            dr = float(np.max(np.abs(rv - app_g)))
                            okr = dr <= 1e-6
                            if not okr and app == 'arvore':
                                app_t = np.array(pg.evaluate(PREV, np.column_stack([X, np.zeros(len(X))]).tolist() if modo == 1 else X.tolist()))
                                if float(np.max(np.abs(app_t - r_executa(e['r'], X, y, X)))) <= 1e-9:
                                    okr = True; dr = -dr  # sinal negativo marca: só entre os pontos (empate)
                        res.append(f'R {"ok" if okr else "DIVERGE"} ({"diferença média" if app in ("boosting", "floresta") else "maior diferença"} {abs(dr):.2e}{"; diferença só entre os pontos: empate entre X₁ e X₂, mesmo ajuste nos pontos de treino" if dr < 0 else ""}{"; gbm e randomForest: parecido, não igual" if app in ("boosting", "floresta") else ""})')
                    except Exception as ex:
                        okr = False
                        res.append('R ERRO: ' + str(ex)[-300:].replace('\n', ' '))
                    if erros: res.append(f'erros no console: {erros}')
                    if not (ok_py and okr) or erros: falhas += 1
                    linhas.append(f'{nome}: ' + '; '.join(res))
                except Exception:
                    falhas += 1
                    linhas.append(f'{nome}: EXCEÇÃO {traceback.format_exc()[-500:]}')
                pg.close()
        b.close()
    srv.shutdown()
    relatorio('t_e2e', [f'ponta a ponta: {sum(len(v) for v in ESTADOS.values())} estados, {falhas} com problema'] + linhas)


# ------------------------------------------------------------------ interação aleatória
RUIM = re.compile(r'NaN|undefined|Infinity|null')
CHECA = """() => { const txt = ['say', 'v-data', 'v-title', 'pill', 'code-r', 'code-py', 'v3d-note'].map(i => document.getElementById(i)?.innerText || '').join(' ') + ' ' + document.querySelector('aside').innerText + ' ' + (document.getElementById('cv3d')?.getAttribute('aria-label') || '');
  const fora = [...document.querySelectorAll('#layer-points .dot')].filter(d => { const x = +d.getAttribute('cx'), y = +d.getAttribute('cy'); return !(x >= 54 && x <= 579 && y >= 8 && y <= 560); }).length;
  return { txt, fora, sw: document.documentElement.scrollWidth, n: window.__regr.state.points.length }; }"""


def aleatorio(n=250, semente=2026):
    rnd = random.Random(semente)
    linhas, total = [], 0
    srv = servidor()
    with sync_playwright() as p:
        b = novo_navegador(p)
        for app in APPS:
            pg = b.new_page(viewport={'width': 1280, 'height': 900})
            erros = []
            coleta_erros(pg, erros)
            pg.goto(f'{BASE}/{app}.html')
            espera_quadro(pg)
            prob = []
            for k in range(n):
                acao = rnd.choice(['modo', 'dados', 'arrasta', 'arrasta', 'adiciona', 'remove', 'param', 'param', 'check', 'tecla', 'mouse', 'ferramenta', 'gira3d', 'viesvar'])
                try:
                    if acao == 'modo':
                        pg.click(f'#vseg button[data-mode="{rnd.choice([1, 2])}"]')
                    elif acao == 'dados':
                        ops = pg.evaluate("() => [...document.querySelectorAll('#dataset option')].map(o => o.value)")
                        pg.select_option('#dataset', rnd.choice(ops))
                    elif acao in ('arrasta', 'adiciona', 'remove'):
                        pts = pg.evaluate("() => window.__regr.state.points.map(p => [p.x, p.y])")
                        if acao == 'adiciona': pg.click('#t-add')
                        if acao == 'remove': pg.click('#t-rem')
                        if acao != 'adiciona' and pts:
                            q = rnd.choice(pts)
                            arrasta(pg, tela(*q), tela(rnd.uniform(-2, 4), rnd.uniform(-2, 4)), 2)
                        else:
                            pg.mouse.click(*[v for v in pg_coords(pg, tela(rnd.uniform(-1.6, 3.4), rnd.uniform(-1.6, 3.6)))])
                        pg.click('#t-move')
                    elif acao == 'param':
                        ids = pg.evaluate("() => [...document.querySelectorAll('.params input[type=number], .datarow input[type=number]')].filter(e => e.offsetParent && !e.disabled).map(e => [e.id, e.min, e.max])")
                        if ids:
                            i, lo, hi = rnd.choice(ids)
                            lo = float(lo or 0); hi = float(hi or 10)
                            v = rnd.choice([lo, hi, rnd.uniform(lo, hi), round(rnd.uniform(lo, hi))])
                            pg.fill('#' + i, str(v))
                        segs = pg.evaluate("() => [...document.querySelectorAll('.params .seg button')].filter(e => e.offsetParent).map((e, k) => k)")
                        if segs and rnd.random() < .5:
                            pg.locator('.params .seg button >> visible=true').nth(rnd.choice(segs)).click()
                    elif acao == 'check':
                        cs = pg.evaluate("() => [...document.querySelectorAll('input[type=checkbox]')].filter(e => e.offsetParent && !e.disabled).map(e => e.id)")
                        if cs: pg.click('#' + rnd.choice(cs))
                    elif acao == 'tecla':
                        pg.locator('#layer-points .pt').first.focus()
                        pg.keyboard.press(rnd.choice(['ArrowUp', 'ArrowLeft', 'Shift+ArrowRight', 'PageUp', 'PageDown', '+', '-', 'Delete']))
                    elif acao == 'mouse':
                        box = pg.locator('#plot').bounding_box()
                        pg.mouse.move(box['x'] + rnd.uniform(0, box['width']), box['y'] + rnd.uniform(0, box['height']))
                    elif acao == 'ferramenta':
                        pg.locator('#tools button >> visible=true').nth(rnd.randrange(3)).click()
                        pg.click('#t-move')
                    elif acao == 'gira3d':
                        # Gráfico 3D (duas variáveis): arrasta para girar, ou setas com o foco nele
                        if pg.locator('#cv3d').is_visible():
                            c = pg.locator('#cv3d')
                            c.scroll_into_view_if_needed()
                            bx = c.bounding_box()
                            if rnd.random() < .5:
                                x0, y0 = bx['x'] + bx['width'] * rnd.uniform(.2, .8), bx['y'] + bx['height'] * rnd.uniform(.2, .8)
                                pg.mouse.move(x0, y0); pg.mouse.down()
                                pg.mouse.move(x0 + rnd.uniform(-200, 200), y0 + rnd.uniform(-150, 150), steps=3); pg.mouse.up()
                            else:
                                c.focus(); pg.keyboard.press(rnd.choice(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'Shift+ArrowDown']))
                            pg.mouse.move(2, 2)
                    elif acao == 'viesvar':
                        # Gráfico de viés e variância: clicar escolhe o parâmetro (se já foi calculado)
                        if pg.locator('#v-bv svg').count() and pg.locator('#v-bv').is_visible():
                            v = pg.locator('#v-bv')
                            v.scroll_into_view_if_needed()
                            bx = v.bounding_box()
                            pg.mouse.click(bx['x'] + bx['width'] * rnd.uniform(.15, .8), bx['y'] + bx['height'] * .5)
                            pg.mouse.move(2, 2)
                    espera_quadro(pg)
                    c = pg.evaluate(CHECA)
                    total += 1
                    m = RUIM.search(c['txt'])
                    if m: prob.append(f'{k} {acao}: texto com "{m.group(0)}" perto de "{c["txt"][max(0, m.start() - 60):m.start() + 20]}"')
                    if c['fora']: prob.append(f'{k} {acao}: {c["fora"]} pontos fora do gráfico')
                    if c['sw'] > 1280: prob.append(f'{k} {acao}: rolagem horizontal ({c["sw"]} px)')
                except Exception as ex:
                    prob.append(f'{k} {acao}: exceção {str(ex)[:200]}')
            if erros: prob.append(f'erros no console: {erros[:5]}')
            linhas.append(f'{app}: {n} ações, {len(prob)} problemas' + (': ' + ' | '.join(prob[:8]) if prob else ''))
            pg.close()
        b.close()
    srv.shutdown()
    relatorio('t_aleatorio', [f'interação aleatória: {total} ações verificadas'] + linhas)


def pg_coords(pg, svgxy):
    box = pg.locator('#plot').bounding_box()
    f = box['width'] / 602
    return box['x'] + svgxy[0] * f, box['y'] + svgxy[1] * f


# ------------------------------------------------------------------ acessibilidade
def axe():
    linhas = []
    srv = servidor()
    ax = Axe()
    with sync_playwright() as p:
        b = novo_navegador(p)
        for pag in ['index'] + APPS:
            for tema in ['light', 'dark']:
                for modo in ([1, 2] if pag != 'index' else [0]):
                    pg = b.new_page(viewport={'width': 1280, 'height': 900})
                    pg.add_init_script(f"try {{ localStorage.setItem('svm-theme', '{tema}'); }} catch (e) {{}}")
                    pg.goto(f'{BASE}/{pag}.html')
                    espera_quadro(pg)
                    if modo:
                        pg.click(f'#vseg button[data-mode="{modo}"]'); pg.check('#o-test'); espera_quadro(pg)
                    r = ax.run(pg)
                    v = r.response['violations']
                    linhas.append(f'{pag} {tema}{f" modo {modo}" if modo else ""}: {len(v)} violações' + (': ' + '; '.join(f"{x['id']} ({len(x['nodes'])})" for x in v) if v else ''))
                    pg.close()
        b.close()
    srv.shutdown()
    n = sum(1 for l in linhas if ': 0 violações' not in l)
    relatorio('t_axe', [f'axe-core: {len(linhas)} páginas e estados, {n} com violações'] + linhas)


# ------------------------------------------------------------------ contextos
def contextos():
    linhas = []
    srv = servidor()
    with sync_playwright() as p:
        b = novo_navegador(p)
        for origem in ['http', 'file']:
            base = BASE if origem == 'http' else (PROJ.as_uri())
            for cel in [False, True]:
                for tema in ['light', 'dark']:
                    vp = {'width': 390, 'height': 844} if cel else {'width': 1280, 'height': 900}
                    ctx = b.new_context(viewport=vp, has_touch=cel, is_mobile=cel)
                    pg = ctx.new_page()
                    erros = []
                    coleta_erros(pg, erros)
                    pg.add_init_script(f"try {{ localStorage.setItem('svm-theme', '{tema}'); }} catch (e) {{}}")
                    prob = []
                    # Dentro das guias: cada app carrega no iframe e recebe o tema
                    pg.goto(f'{base}/index.html')
                    espera_quadro(pg)
                    for app in APPS:
                        pg.click(f'#tab-{app}')
                        pg.wait_for_timeout(500)
                        fr = pg.frame_locator(f'#frame-{app}')
                        try:
                            n = fr.locator('#layer-points .pt').count()
                            t = pg.evaluate(f"() => document.getElementById('frame-{app}').contentDocument?.documentElement.getAttribute('data-theme')") if origem == 'http' else 'n/d'
                            if n < 2: prob.append(f'{app} nas guias: {n} pontos')
                            if origem == 'http' and t != tema: prob.append(f'{app}: tema {t} no iframe, esperado {tema}')
                        except Exception as ex:
                            prob.append(f'{app} nas guias: {str(ex)[:120]}')
                    sw = pg.evaluate("() => document.documentElement.scrollWidth")
                    if sw > vp['width']: prob.append(f'índice com rolagem horizontal {sw}')
                    # Botão de tema troca o tema dentro do app visível
                    pg.click('#theme'); pg.wait_for_timeout(200)
                    if origem == 'http':
                        t2 = pg.evaluate("() => document.querySelector('iframe:not([hidden])').contentDocument.documentElement.getAttribute('data-theme')")
                        if t2 == tema: prob.append('botão de tema não chegou ao app')
                    # Cada app sozinho: rolagem horizontal e arraste (toque simulado no celular)
                    for app in APPS:
                        pg.goto(f'{base}/{app}.html')
                        espera_quadro(pg)
                        for modo in (1, 2):
                            pg.click(f'#vseg button[data-mode="{modo}"]'); espera_quadro(pg)
                            sw = pg.evaluate("() => document.documentElement.scrollWidth")
                            if sw > vp['width']: prob.append(f'{app} modo {modo}: rolagem horizontal {sw} > {vp["width"]}')
                            pg.locator('#plot').scroll_into_view_if_needed(); espera_quadro(pg)
                            antes = pg.evaluate("() => { const p = window.__regr.state.points[0]; return [p.x, p.y]; }")
                            de = pg_coords(pg, tela(*antes)); para = (de[0] + 25, de[1] - 20)
                            if not (0 < de[1] < vp['height']): prob.append(f'{app} modo {modo}: ponto fora da tela ({de[1]:.0f} px)')
                            if cel:
                                cdp = ctx.new_cdp_session(pg)
                                for tipo, (x, y) in [('touchStart', de), ('touchMove', ((de[0] + para[0]) / 2, (de[1] + para[1]) / 2)), ('touchMove', para), ('touchEnd', para)]:
                                    cdp.send('Input.dispatchTouchEvent', {'type': tipo, 'touchPoints': [] if tipo == 'touchEnd' else [{'x': x, 'y': y}]})
                                    pg.wait_for_timeout(40)
                            else:
                                pg.mouse.move(*de); pg.mouse.down(); pg.mouse.move(*para, steps=4); pg.mouse.up()
                            espera_quadro(pg)
                            depois = pg.evaluate("() => { const p = window.__regr.state.points[0]; return [p.x, p.y]; }")
                            if abs(depois[0] - antes[0]) < 0.05: prob.append(f'{app} modo {modo}: o arraste não moveu o ponto')
                    if erros: prob.append(f'erros no console: {erros[:4]}')
                    linhas.append(f'{origem} {"celular 390 px, toque" if cel else "desktop 1280 px"} {tema}: {len(prob)} problemas' + (': ' + ' | '.join(prob[:8]) if prob else ''))
                    ctx.close()
        b.close()
    srv.shutdown()
    relatorio('t_contextos', linhas)


# ------------------------------------------------------------------ desempenho
def desempenho():
    linhas = []
    srv = servidor()
    with sync_playwright() as p:
        b = novo_navegador(p)
        casos = [('linear', [], 'grau 10', [('fill', '#p-d-n', '10')]), ('penalizada', [], '', []), ('knn', [], '', []), ('arvore', [], '', []),
                 ('floresta', [], 'B = 100', []), ('floresta', [], 'B = 500', [('fill', '#p-b-n', '500')]),
                 ('boosting', [], 'B = 500, d = 1', []), ('boosting', [], 'B = 1000, d = 5', [('fill', '#p-nt-n', '1000'), ('fill', '#p-d-n', '5')])]
        for app, _, rot, acoes in casos:
            for modo in (1, 2):
                pg = b.new_page(viewport={'width': 1280, 'height': 900})
                pg.goto(f'{BASE}/{app}.html'); espera_quadro(pg)
                pg.click(f'#vseg button[data-mode="{modo}"]'); espera_quadro(pg)
                for a in acoes:
                    if pg.locator(a[1]).is_visible(): pg.fill(a[1], a[2]); espera_quadro(pg)
                # Com duas variáveis, o gráfico 3D na tela (só é desenhado quando aparece): entra na conta do quadro.
                # Durante o arraste ele usa ladrilhos maiores; o desenho fino (0,22 s depois) é medido à parte, abaixo.
                if modo == 2:
                    pg.locator('#s-3d').scroll_into_view_if_needed(); pg.wait_for_timeout(150)
                # n = 60 pontos e conjunto de teste ligado; mede o redesenho completo com um ponto mudando de lugar
                for N in (30, 60):
                  ms = pg.evaluate("""(N) => { const r = window.__regr, s = r.state; const ops = [...document.querySelectorAll('#dataset option')].map(o => o.value);
                  const pts = r.loadData(s.mode, ops.find(v => r.DATA[s.mode][v].n === 60) || ops[0], s.sigma).slice(0, N);
                  while (pts.length < N) pts.push({ ...pts[pts.length % 10], x: pts[pts.length % 10].x + .013*pts.length });
                  r.pontos(pts); r.desenha(); const t = []; for (let k = 0; k < 12; k++) { s.points[0].x += .01; const t0 = performance.now(); r.desenha(); t.push(performance.now() - t0); }
                  t.sort((a, b) => a - b); return [t[6], t[11]]; }""", N)
                  linhas.append(f'{app} {rot} modo {modo}, n = {N}: mediana {ms[0]:.1f} ms, pior {ms[1]:.1f} ms por quadro{"  (acima de 16 ms)" if ms[0] > 16 else ""}')
                  if modo == 2:
                      t3 = []
                      for _ in range(5):
                          pg.wait_for_timeout(500)  # mais de 0,2 s depois do último desenho: ladrilhos finos
                          t3.append(pg.evaluate("() => { const t0 = performance.now(); window.__regr.draw3d(); return performance.now() - t0; }"))
                      linhas.append(f'    3D completo (parado, desenho fino), n = {N}: mediana {sorted(t3)[2]:.1f} ms')
                pg.close()
        b.close()
    srv.shutdown()
    relatorio('t_desempenho', linhas)


# ------------------------------------------------------------------ capturas
def capturas():
    from PIL import Image
    pasta = RES / 'capturas'
    pasta.mkdir(exist_ok=True)
    srv = servidor()
    feitas = []
    with sync_playwright() as p:
        b = novo_navegador(p)
        for app in APPS:
            for modo in (1, 2):
                for tema, cel in [('light', False), ('dark', False), ('light', True)]:
                    vp = {'width': 390, 'height': 844} if cel else {'width': 1300, 'height': 900}
                    pg = b.new_page(viewport=vp, has_touch=cel, is_mobile=cel)
                    pg.add_init_script(f"try {{ localStorage.setItem('svm-theme', '{tema}'); }} catch (e) {{}}")
                    pg.goto(f'{BASE}/{app}.html'); espera_quadro(pg)
                    pg.click(f'#vseg button[data-mode="{modo}"]'); pg.check('#o-test'); espera_quadro(pg)
                    arq = pasta / f'cap_{app}_{modo}_{tema}{"_cel" if cel else ""}.png'
                    pg.screenshot(path=str(arq), full_page=cel)
                    feitas.append(arq)
                    # Gráfico 3D (duas variáveis) e viés e variância, depois de calculado
                    if modo == 2:
                        pg.check('#o-truef')
                        pg.locator('#s-3d').scroll_into_view_if_needed(); pg.wait_for_timeout(600)
                        f3 = pasta / f'cap3d_{app}_{tema}{"_cel" if cel else ""}.png'
                        pg.locator('#s-3d').screenshot(path=str(f3)); feitas.append(f3)
                    if pg.locator('#s-bv').is_visible():
                        pg.locator('#s-bv').scroll_into_view_if_needed()
                        pg.wait_for_function('() => !!window.__regr.bv()', timeout=120000); espera_quadro(pg)
                        fb = pasta / f'capbv_{app}_{modo}_{tema}{"_cel" if cel else ""}.png'
                        pg.locator('#s-bv').screenshot(path=str(fb)); feitas.append(fb)
                    pg.close()
        # Mosaico dos conjuntos de dados (knn), nos dois modos
        pg = b.new_page(viewport={'width': 1300, 'height': 900})
        pg.goto(f'{BASE}/knn.html'); espera_quadro(pg)
        imgs = []
        for modo in (1, 2):
            pg.click(f'#vseg button[data-mode="{modo}"]'); espera_quadro(pg)
            for v in pg.evaluate("() => [...document.querySelectorAll('#dataset option')].map(o => o.value)"):
                pg.select_option('#dataset', v); pg.check('#o-truef'); espera_quadro(pg)
                f = pasta / f'mosaico_{modo}_{v}.png'
                pg.locator('.plotwrap').screenshot(path=str(f)); imgs.append(f)
        pg.close()
        b.close()
    srv.shutdown()
    ims = [Image.open(f).resize((300, 313)) for f in imgs]
    cols = 7
    M = Image.new('RGB', (cols * 300, ((len(ims) + cols - 1) // cols) * 313), 'white')
    for k, im in enumerate(ims):
        M.paste(im, ((k % cols) * 300, (k // cols) * 313))
    M.save(pasta / 'mosaico.png')
    relatorio('t_capturas', [f'{len(feitas)} capturas e mosaico com {len(ims)} conjuntos em {pasta}'])


if __name__ == '__main__':
    qual = sys.argv[1] if len(sys.argv) > 1 else 'todos'
    for nome, f in [('e2e', e2e), ('aleatorio', aleatorio), ('axe', axe), ('contextos', contextos), ('desempenho', desempenho), ('capturas', capturas)]:
        if qual in (nome, 'todos'):
            f()
