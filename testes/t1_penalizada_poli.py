"""Penalizada com polinômio (uma variável, colunas x, x², …, x^d padronizadas): o solver de conjunto ativo do app
(Solver.solveFS, p > 2) contra scikit-learn (make_pipeline(PolynomialFeatures, StandardScaler, Ridge/Lasso/ElasticNet)),
o QP do cvxopt e o glmnet do R, em 1000 conjuntos (os 500 de uma variável do dados.py, cada um com dois graus).
Com colunas quase dependentes (grau alto), os coeficientes podem ser mal determinados; por isso a comparação principal é
a do valor do objetivo e a dos valores ajustados (únicos no ridge, no lasso e na elastic net), além dos coeficientes.
Também confere o solveFS contra a enumeração exata nos casos com p ≤ 2.
Uso: python t1_penalizada_poli.py
"""
import os
import subprocess
import tempfile
import warnings

import numpy as np
from cvxopt import matrix, solvers
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from comum import RS, relatorio, roda_js
from dados import gera
from t1_penalizada import objetivo, padroniza, qp

warnings.filterwarnings('ignore')
solvers.options['show_progress'] = False

JS = r"""
const X = caso.d > 1 ? Solver.powers(caso.X, caso.d) : caso.X;
const pr = Solver.prep(X, caso.y, caso.std);
const r = Solver.solveFS(pr.G, pr.c, caso.lam, caso.alpha, null);
// caminho com partida quente, como no app: do maior λ ao atual
let warm = null;
for (const t of [4, 3, 2, 1, 0]) { if (10**t <= caso.lam) break; const q = Solver.solveFS(pr.G, pr.c, 10**t, caso.alpha, warm); if (q) warm = q.b; }
const rw = Solver.solveFS(pr.G, pr.c, caso.lam, caso.alpha, warm);
const en = pr.p <= 2 ? Solver.solve(pr.G, pr.c, caso.lam, caso.alpha) : null;
return { b: r ? Array.from(r.b) : null, kkt: r ? r.kkt : null, cd: r ? !!r.cd : null, bw: rw ? Array.from(rw.b) : null, en: en ? Array.from(en) : null };
"""


def sk(x, y, d, std, lam, a):
    n = len(y)
    if lam == 0:
        est = LinearRegression()
    elif a == 0:
        est = Ridge(alpha=lam)
    elif a == 1:
        est = Lasso(alpha=lam / (2 * n), max_iter=1_000_000, tol=1e-14)
    else:
        est = ElasticNet(alpha=lam * (2 - a) / (2 * n), l1_ratio=a / (2 - a), max_iter=1_000_000, tol=1e-14)
    pf = PolynomialFeatures(degree=d, include_bias=False)
    m = make_pipeline(pf, StandardScaler(), est) if std else make_pipeline(pf, est)
    m.fit(x.reshape(-1, 1), y)
    return m.predict(x.reshape(-1, 1)), m[-1].coef_


def r_glmnet(casos):
    """Valores ajustados do glmnet com o código mostrado na tela (poly → padronização com divisor n → glmnet)."""
    with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as dd:
        linhas = []
        for k, c in enumerate(casos):
            n, a, lam = len(c['y']), c['alpha'], c['lam']
            sy = float(np.std(c['y']))
            a1, a2 = lam * a / (2 * n), lam * (1 - a) / n
            gl = a1 + sy * a2
            ga = a1 / gl if gl > 0 else 1.0
            for i in range(n):
                linhas.append(','.join(map(str, [k, c['d'], int(c['std']), ga, gl, float(c['X'][i][0]), float(c['y'][i])])))
        arq = os.path.join(dd, 'g.csv')
        with open(arq, 'w') as f:
            f.write('\n'.join(linhas) + '\n')
        scr = r"""
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); suppressPackageStartupMessages(library(glmnet))
A <- read.csv(commandArgs(TRUE)[1], header = FALSE)
for (k in unique(A$V1)) { B <- A[A$V1 == k, ]; x <- B$V6; y <- B$V7; d <- B$V2[1]
  X <- poly(x, d, raw = TRUE); n <- nrow(X)
  Xs <- if (B$V3[1] == 1) scale(X) * sqrt(n / (n - 1)) else as.matrix(X)
  m <- tryCatch(glmnet(Xs, y, alpha = B$V4[1], lambda = B$V5[1], standardize = FALSE, control = list(thresh = 1e-14, maxit = 1e7)), error = function(e) NULL)
  if (is.null(m)) cat(k, "ERRO\n") else cat(k, sprintf("%.17g", as.numeric(predict(m, Xs))), "\n") }
"""
        rf = os.path.join(dd, 'g.R')
        with open(rf, 'w') as f:
            f.write(scr)
        r = subprocess.run([RS, rf, arq], capture_output=True, text=True)
    res = {}
    for ln in r.stdout.strip().splitlines():
        v = ln.split()
        res[int(v[0])] = None if v[1] == 'ERRO' else np.array([float(t) for t in v[1:]])
    return res, r.stderr[-500:]


def main():
    rng = np.random.default_rng(11)
    base = [c for c in gera() if len(c['X'][0]) == 1]
    casos = []
    for k, c in enumerate(base):
        for rep in range(2):
            c2 = dict(c)
            c2['d'] = int(rng.integers(2, 11)) if rep == 0 else int(rng.choice([3, 6, 10]))
            c2['alpha'] = [0.0, 1.0, 0.25, 0.5, 0.75, 1.0][(2 * k + rep) % 6]
            c2['lam'] = float(10 ** rng.uniform(-2, 4))
            c2['std'] = (2 * k + rep) % 4 != 3
            casos.append(c2)
    out, ms = roda_js('penalizada', JS, casos)
    F = {k: [] for k in ['js', 'kkt', 'cd', 'obj_qp', 'obj_sk', 'fit_sk', 'fit_qp', 'quente', 'enum', 'glmnet']}
    M = {k: 0.0 for k in F}
    for i, (c, o) in enumerate(zip(casos, out)):
        if 'erro' in o or o.get('b') is None:
            F['js'].append((i, o.get('erro', '')[:200]))
            continue
        x, y, d, lam, a = np.array(c['X'], float)[:, 0], np.array(c['y'], float), c['d'], c['lam'], c['alpha']
        P = np.column_stack([x ** k for k in range(1, d + 1)])
        Xs, mx, sd = padroniza(P, c['std'])
        b = np.array(o['b'])
        M['kkt'] = max(M['kkt'], o['kkt'])
        if o['kkt'] > 1e-7: F['kkt'].append((i, o['kkt']))
        if o['cd']: F['cd'].append(i)
        fa = y.mean() + (Xs - Xs.mean(0)) @ b
        oa = objetivo(Xs, y, b, lam, a)
        bq, st = qp(Xs, y, lam, a)
        oq = objetivo(Xs, y, bq, lam, a)
        pred, cs = sk(x, y, d, c['std'], lam, a)
        os_ = objetivo(Xs, y, cs, lam, a)
        esc = 1 + abs(min(oq, os_))
        e = (oa - oq) / esc
        M['obj_qp'] = max(M['obj_qp'], e)
        if e > 1e-9: F['obj_qp'].append((i, e, st))
        e = (oa - os_) / esc
        M['obj_sk'] = max(M['obj_sk'], e)
        if e > 1e-9: F['obj_sk'].append((i, e))
        sc = 1 + float(np.max(np.abs(y)))
        e = float(np.max(np.abs(fa - pred))) / sc
        M['fit_sk'] = max(M['fit_sk'], e)
        if e > 1e-6: F['fit_sk'].append((i, round(e, 8), a, d, round(lam, 4), round((oa - os_) / esc, 12)))
        fq = y.mean() + (Xs - Xs.mean(0)) @ bq
        e = float(np.max(np.abs(fa - fq))) / sc
        M['fit_qp'] = max(M['fit_qp'], e)
        if e > 1e-6: F['fit_qp'].append((i, round(e, 8), round((oa - oq) / esc, 12)))
        bw = np.array(o['bw'])
        fw = y.mean() + (Xs - Xs.mean(0)) @ bw
        e = float(np.max(np.abs(fa - fw))) / sc
        M['quente'] = max(M['quente'], e)
        if e > 1e-8: F['quente'].append((i, e))
        if o['en'] is not None:
            e = float(np.max(np.abs(np.array(o['en']) - b))) / (1 + float(np.max(np.abs(b))))
            M['enum'] = max(M['enum'], e)
            if e > 1e-8: F['enum'].append((i, e))
    sel = [i for i in range(len(casos)) if casos[i]['lam'] > 0 and np.ptp(np.array(casos[i]['y'])) > 0 and len(set(np.array(casos[i]['X'])[:, 0])) > 2]
    G, err = r_glmnet([casos[i] for i in sel])
    nerr = 0
    for k, i in enumerate(sel):
        g, o = G.get(k), out[i]
        if g is None or o.get('b') is None:
            nerr += 1
            continue
        c = casos[i]
        x, y = np.array(c['X'], float)[:, 0], np.array(c['y'], float)
        P = np.column_stack([x ** q for q in range(1, c['d'] + 1)])
        Xs, _, _ = padroniza(P, c['std'])
        fa = y.mean() + (Xs - Xs.mean(0)) @ np.array(o['b'])
        e = float(np.max(np.abs(fa - g))) / (1 + float(np.max(np.abs(y))))
        M['glmnet'] = max(M['glmnet'], e)
        if e > 1e-5: F['glmnet'].append((i, round(e, 6), c['alpha'], c['d']))
    graus = np.bincount([c['d'] for c in casos], minlength=11)[2:]
    linhas = [f'penalizada.html, polinômio: conjunto ativo (solveFS) em {len(casos)} conjuntos de uma variável ({ms:.0f} ms no Node); graus 2–10 (quantos de cada: {list(graus)}); λ log-uniforme em [0,01; 10⁴]; ridge, lasso e elastic net; com e sem padronizar',
              f'- sem solução no JS (ou erro): {len(F["js"])}' + (f'; {F["js"][:3]}' if F['js'] else ''),
              f'- violação das condições KKT relatada pelo app > 1e-7 (relativa): {len(F["kkt"])}; maior {M["kkt"]:.2e}' + (f'; exemplos {F["kkt"][:4]}' if F['kkt'] else ''),
              f'- casos que caíram na descida por coordenadas (reserva): {len(F["cd"])}',
              f'- objetivo do app maior que o do QP do cvxopt (relativo > 1e-9): {len(F["obj_qp"])}; maior {M["obj_qp"]:.2e}' + (f'; exemplos {F["obj_qp"][:4]}' if F['obj_qp'] else ''),
              f'- objetivo do app maior que o do scikit-learn (relativo > 1e-9): {len(F["obj_sk"])}; maior {M["obj_sk"]:.2e}' + (f'; exemplos {F["obj_sk"][:4]}' if F['obj_sk'] else ''),
              f'- valores ajustados contra o scikit-learn (pipeline com PolynomialFeatures), > 1e-6: {len(F["fit_sk"])}; maior {M["fit_sk"]:.2e}' + (f'; exemplos (caso, dif., α, d, λ, objetivo app − sk) {F["fit_sk"][:5]}' if F['fit_sk'] else ''),
              f'- valores ajustados contra o QP do cvxopt, > 1e-6: {len(F["fit_qp"])}; maior {M["fit_qp"]:.2e}' + (f'; exemplos (caso, dif., objetivo app − QP) {F["fit_qp"][:5]}' if F['fit_qp'] else ''),
              f'- partida quente (como no caminho do app) contra partida do zero, valores ajustados > 1e-8: {len(F["quente"])}; maior {M["quente"]:.2e}',
              f'- solveFS contra a enumeração exata (grau 2, p = 2): {len(F["enum"])} divergências > 1e-8; maior {M["enum"]:.2e}',
              f'- valores ajustados contra o glmnet ({len(sel)} casos, código da tela), > 1e-5: {len(F["glmnet"])}; maior {M["glmnet"]:.2e}; erros no R: {nerr}' + (f'; exemplos (caso, dif., α, d) {F["glmnet"][:6]}' if F['glmnet'] else '')]
    if err.strip() and nerr:
        linhas.append('stderr do R: ' + err)
    relatorio('t1_penalizada_poli', linhas)


if __name__ == '__main__':
    main()
