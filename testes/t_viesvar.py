"""Viés² e variância (gráfico "Viés e variância, como na Fig. 2.12" e Variabilidade): os números do app contra a mesma conta
refeita aqui com scikit-learn e numpy, usando as amostras de treino e os pontos x₀ que o app usa (lidos do gancho __regr).
Também confere: MSE esperado = viés² + variância + σ² ≥ σ²; reta com σ = 0 e grau 1 → viés² = variância = 0; e que a
Variabilidade (uma variável) dá os mesmos números do gráfico no valor atual do parâmetro.
Uso: python t_viesvar.py
"""
import numpy as np
from playwright.sync_api import sync_playwright
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

from base import BASE, espera_quadro, novo_navegador, servidor
from comum import relatorio


def ajusta(app, v, X, y, P, Xe):
    if app == 'knn':
        return KNeighborsRegressor(n_neighbors=min(v, len(y)), weights=P['w'], p=P['p'], algorithm='brute').fit(X, y).predict(Xe)
    if app == 'linear':
        if X.shape[1] == 1:
            pf = PolynomialFeatures(v, include_bias=False)
            return LinearRegression().fit(pf.fit_transform(X), y).predict(pf.transform(Xe))
        f = (lambda Z: np.column_stack([Z, Z[:, 0] * Z[:, 1]])) if v else (lambda Z: Z)
        return LinearRegression().fit(f(X), y).predict(f(Xe))
    if app == 'arvore':
        return DecisionTreeRegressor(ccp_alpha=v / len(y), random_state=0).fit(X, y).predict(Xe)
    if app == 'penalizada':  # ridge com grau d (uma variável), padronizado
        pf = PolynomialFeatures(P['d'], include_bias=False)
        return make_pipeline(pf, StandardScaler(), Ridge(alpha=v)).fit(X, y).predict(Xe)


CASOS = [  # app, modo, conjunto, σ (None = padrão), ações extras em JS
    ('knn', 1, 'curva', None, ''),
    ('knn', 2, 'chapeu', None, ''),
    ('knn', 1, 'seno', None, "window.__regr.state.P.w = 'distance'; window.__regr.state.P.p = 1;"),
    ('linear', 1, 'curva', None, ''),
    ('linear', 2, 'inter', None, ''),
    ('linear', 1, 'reta', 0.0, ''),
    ('arvore', 1, 'degrau', None, ''),
    ('penalizada', 1, 'curva', None, "window.__regr.state.P.pen = 'ridge'; window.__regr.state.P.d = 6;"),
]


def main():
    srv = servidor()
    linhas = ['Viés² e variância: app contra a conta refeita com scikit-learn/numpy (mesmas 40 amostras e mesmos x₀)']
    problemas = 0
    with sync_playwright() as p:
        b = novo_navegador(p)
        for app, modo, ds, sg, js in CASOS:
            pg = b.new_page(viewport={'width': 1300, 'height': 1600})
            pg.goto(f'{BASE}/{app}.html')
            espera_quadro(pg)
            pg.click(f'#vseg button[data-mode="{modo}"]')
            pg.select_option('#dataset', ds)
            if sg is not None:
                pg.fill('#p-noise-n', str(sg))
            if js:
                pg.evaluate(f'() => {{ {js} window.__regr.api.render(); }}')
            pg.check('#o-var') if modo == 1 else None
            pg.wait_for_function('() => window.__regr.bv() && (window.__regr.state.mode === 2 || window.__regr.vari())', timeout=120000)
            d = pg.evaluate("""() => { const R = window.__regr, s = R.state, E = R.evalSet(), bv = R.bv();
              const vi = R.vari();
              return { E: E.X, fx: Array.from(E.fx), vals: bv.vals, b2: bv.b2, v: bv.v, mse: bv.mse, s2: bv.s2, P: s.P, sigma: s.sigma,
                       cur: App.biasVar.cur(s.P, s.model, R.api), vari: vi ? { b2: vi.b2, v: vi.v } : null }; }""")
            Xe, fx = np.array(d['E'], float), np.array(d['fx'])
            preds = {k: [] for k in range(len(d['vals']))}
            am = pg.evaluate("""() => { const R = window.__regr, s = R.state, out = []; for (let r = 1; r <= 40; r++) out.push(R.loadData(s.mode, s.dataset, s.sigma, 2000 + r).map(q => s.mode === 1 ? [q.x, q.y] : [q.x, q.y, q.v])); return out; }""")
            for A in am:
                A = np.array(A, float)
                X, y = A[:, :modo], A[:, modo]
                for k, v in enumerate(d['vals']):
                    preds[k].append(ajusta(app, v, X, y, d['P'], Xe))
            b2 = np.array([np.mean((np.mean(preds[k], 0) - fx) ** 2) for k in preds])
            va = np.array([np.mean(np.var(preds[k], 0, ddof=1)) for k in preds])
            eb = float(np.max(np.abs(b2 - np.array(d['b2'])) / (1e-9 + np.maximum(b2, 1e-6))))
            ev = float(np.max(np.abs(va - np.array(d['v'])) / (1e-9 + np.maximum(va, 1e-6))))
            ok = eb < 1e-6 and ev < 1e-6
            extra = []
            mse = np.array(d['mse'])
            if np.any(mse < d['s2'] - 1e-12):
                ok = False
                extra.append('MSE esperado < σ²')
            if sg == 0.0:
                z = float(max(np.max(np.abs(d['b2'])), 0))
                extra.append(f'σ = 0: viés² e variância no grau 1 = {d["b2"][0]:.1e} e {d["v"][0]:.1e}')
                if d['b2'][0] > 1e-12 or d['v'][0] > 1e-12:  # zero a menos de arredondamento
                    ok = False
            if d['vari'] is not None and d['cur'] in d['vals']:
                k = d['vals'].index(d['cur'])
                e = max(abs(d['vari']['b2'] - d['b2'][k]), abs(d['vari']['v'] - d['v'][k]))
                extra.append(f'Variabilidade = gráfico no valor atual: dif. {e:.1e}')
                if e > 1e-12:
                    ok = False
            problemas += not ok
            linhas.append(f'- {app}, {modo} var., {ds}{"" if sg is None else f", σ = {sg}"}: {len(d["vals"])} valores; maior dif. relativa viés² {eb:.1e}, variância {ev:.1e}; {"ok" if ok else "PROBLEMA"}' + (f' ({"; ".join(extra)})' if extra else ''))
            pg.close()
        b.close()
    srv.shutdown()
    linhas.append(f'Casos com problema: {problemas}')
    relatorio('t_viesvar', linhas)


if __name__ == '__main__':
    main()
