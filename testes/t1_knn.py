"""Regressão KNN (knn.html): o solver do app contra KNeighborsRegressor (algorithm="brute") e contra FNN::knn.reg e kknn do R, em 1000 conjuntos.
As previsões são comparadas numa grade e nos próprios pontos de treino. Consultas com empate de distância entre o K-ésimo e o (K+1)-ésimo
vizinho são contadas à parte: nelas a escolha do vizinho depende da implementação (veja "Desempates" no app).
Uso: python t1_knn.py
"""
import os
import subprocess
import tempfile
import warnings

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from comum import RS, relatorio, roda_js
from dados import gera, grade_teste

warnings.filterwarnings('ignore')

JS = r"""
const K = Math.min(caso.K, caso.X.length), f = Solver.makeF(caso.X, caso.y, K, caso.p, caso.w);
return { grade: caso.G.map(g => f(g[0], g[1])), treino: caso.X.map(x => f(x[0], x[1])), lento: caso.G.slice(0, 40).map(g => Solver.predict(caso.X, caso.y, g, K, caso.p, caso.w).yhat) };
"""


def empate(X, q, K, p):
    """Há empate entre o K-ésimo e o (K+1)-ésimo vizinho de q?"""
    if K >= len(X):
        return False
    d = np.sort(np.sum(np.abs(X - q) ** p, axis=1))
    return abs(d[K - 1] - d[K]) <= 1e-9 * (1 + d[K])


def r_ref(casos):
    """FNN::knn.reg (pesos iguais, p = 2) e kknn (distance = p, kernel rectangular ou inv) nos pontos da grade."""
    with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as d:
        L, Gl = [], []
        for k, c in enumerate(casos):
            X = np.asarray(c['X']); pdim = X.shape[1]
            for i in range(len(X)):
                L.append(','.join(map(str, [k, c['K'], c['p'], int(c['w'] == 'distance'), float(X[i, 0]), float(X[i, 1]) if pdim > 1 else 'NA', float(c['y'][i])])))
            for g in c['G'][:60]:
                Gl.append(','.join(map(str, [k, float(g[0]), float(g[1]) if pdim > 1 else 'NA'])))
        a, b = os.path.join(d, 'a.csv'), os.path.join(d, 'b.csv')
        open(a, 'w').write('\n'.join(L) + '\n'); open(b, 'w').write('\n'.join(Gl) + '\n')
        scr = r"""
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); suppressPackageStartupMessages({library(FNN); library(kknn)})
a <- commandArgs(TRUE); A <- read.csv(a[1], header = FALSE); G <- read.csv(a[2], header = FALSE)
for (k in unique(A$V1)) { B <- A[A$V1 == k, ]; Q <- G[G$V1 == k, ]; two <- !all(is.na(B$V6))
  X <- if (two) data.frame(X1 = B$V5, X2 = B$V6) else data.frame(X1 = B$V5); x0 <- if (two) data.frame(X1 = Q$V2, X2 = Q$V3) else data.frame(X1 = Q$V2)
  dados <- cbind(X, y = B$V7); K <- B$V2[1]; p <- B$V3[1]; inv <- B$V4[1] == 1
  f <- if (!inv && (p == 2 || !two)) tryCatch(knn.reg(train = X, test = x0, y = B$V7, k = K, algorithm = "brute")$pred, error = function(e) rep(NA, nrow(x0))) else rep(NA, nrow(x0))
  g <- tryCatch(kknn(y ~ ., train = dados, test = x0, k = K, distance = if (two) p else 2, kernel = if (inv) "inv" else "rectangular", scale = FALSE)$fitted.values, error = function(e) rep(NA, nrow(x0)))
  cat(k, "F", sprintf("%.17g", f), "\n"); cat(k, "K", sprintf("%.17g", g), "\n") }
"""
        rf = os.path.join(d, 'r.R'); open(rf, 'w').write(scr)
        r = subprocess.run([RS, rf, a, b], capture_output=True, text=True)
    res = {}
    for ln in r.stdout.strip().splitlines():
        v = ln.split()
        res[(int(v[0]), v[1])] = np.array([float(t) if t != 'NA' else np.nan for t in v[2:]])
    return res, r.stderr[-400:]


def main():
    casos = gera()
    for i, c in enumerate(casos):
        n, pdim = len(c['X']), len(c['X'][0])
        c['K'] = [1, 3, 5, 9, n][i % 5]
        c['w'] = ['uniform', 'distance'][(i // 5) % 2]
        c['p'] = [2, 1, 3][(i // 10) % 3]
        G = grade_teste(pdim, 21)
        c['G'] = np.column_stack([G, np.zeros(len(G))]).tolist() if pdim == 1 else G.tolist()
        c['X'] = c['X'] if pdim == 2 else c['X']
    out, ms = roda_js('knn', JS, casos)
    F = {'grade': [], 'treino': [], 'lento': []}
    M = {k: 0.0 for k in F}
    nemp, njs = 0, 0
    for i, (c, o) in enumerate(zip(casos, out)):
        if 'erro' in o:
            njs += 1
            continue
        X, y, pdim = np.asarray(c['X']), np.asarray(c['y']), len(c['X'][0])
        K = min(c['K'], len(X))
        m = KNeighborsRegressor(n_neighbors=K, weights=c['w'], algorithm='brute', p=c['p'], metric='minkowski').fit(X, y)
        G = np.asarray(c['G'])[:, :pdim]
        for nome, Q, val in [('grade', G, o['grade']), ('treino', X, o['treino'])]:
            ref = m.predict(Q)
            for q, a, b in zip(Q, val, ref):
                e = abs(a - b)
                if e > 1e-9:
                    if empate(X, q, K, c['p']) or (c['w'] == 'distance' and np.min(np.sum(np.abs(X - q) ** c['p'], axis=1)) < 1e-12):
                        nemp += 1
                    else:
                        F[nome].append((i, round(e, 6))); M[nome] = max(M[nome], e)
        e = float(np.max(np.abs(np.array(o['lento']) - np.array(o['grade'][:40]))))
        M['lento'] = max(M['lento'], e)
        if e > 1e-12: F['lento'].append((i, e))
    sel = list(range(0, 1000, 5))
    R, err = r_ref([casos[i] for i in sel])
    rr = {'F': [0, 0, 0.0], 'K': [0, 0, 0.0]}  # comparados, divergentes (fora de empates), maior
    for k, i in enumerate(sel):
        c, o = casos[i], out[i]
        X = np.asarray(c['X']); K = min(c['K'], len(X)); G = np.asarray(c['G'])[:60, :X.shape[1]]
        for chave in 'FK':
            v = R.get((k, chave))
            if v is None:
                continue
            for q, a, b in zip(G, o['grade'][:60], v):
                if not np.isfinite(b):
                    continue
                if chave == 'K' and K >= len(X):
                    continue  # kknn não reproduz com k = n (usa o (k+1)-ésimo vizinho)
                rr[chave][0] += 1
                if abs(a - b) > 1e-6 and not empate(X, q, K, c['p']):
                    rr[chave][1] += 1; rr[chave][2] = max(rr[chave][2], abs(a - b))
    linhas = [f'knn.html: solver KNN em {len(casos)} conjuntos ({ms:.0f} ms no Node); K ∈ {{1, 3, 5, 9, n}}, pesos iguais e 1/d, p ∈ {{1, 2, 3}}',
              f'- erros de execução no JS: {njs}',
              f'- grade de consulta contra KNeighborsRegressor: {len(F["grade"])} divergências > 1e-9 fora de empates; maior {M["grade"]:.2e}' + (f'; exemplos {F["grade"][:5]}' if F['grade'] else ''),
              f'- pontos de treino contra KNeighborsRegressor: {len(F["treino"])} divergências > 1e-9 fora de empates; maior {M["treino"]:.2e}' + (f'; exemplos {F["treino"][:5]}' if F['treino'] else ''),
              f'- consultas com empate de distância no K-ésimo vizinho (ou vizinho a distância zero com pesos 1/d) em que app e scikit-learn diferem: {nemp}',
              f'- versão rápida (grade) contra a versão direta do app: maior diferença {M["lento"]:.2e}',
              f'- R, FNN::knn.reg ({len(sel)} conjuntos, pesos iguais e p = 2): {rr["F"][0]} consultas, {rr["F"][1]} divergências > 1e-6 fora de empates, maior {rr["F"][2]:.2e}',
              f'- R, kknn (mesmos conjuntos, K < n): {rr["K"][0]} consultas, {rr["K"][1]} divergências > 1e-6 fora de empates, maior {rr["K"][2]:.2e}']
    relatorio('t1_knn', linhas)


if __name__ == '__main__':
    main()
