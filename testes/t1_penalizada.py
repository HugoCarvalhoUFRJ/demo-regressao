"""Ridge, lasso e elastic net (penalizada.html): o solver exato do app contra scikit-learn, um otimizador genérico (QP do cvxopt)
e o glmnet do R, em 1000 conjuntos. A escala de λ é a do livro (Eqs. (6.5) e (6.7)); a elastic net é RSS + λ[α‖β‖₁ + (1 − α)‖β‖²].
Uso: python t1_penalizada.py
"""
import os
import subprocess
import tempfile
import warnings

import numpy as np
from cvxopt import matrix, solvers
from sklearn.linear_model import ElasticNet, Lasso, Ridge

from comum import RS, relatorio, roda_js
from dados import gera

warnings.filterwarnings('ignore')
solvers.options['show_progress'] = False
solvers.options['abstol'] = solvers.options['reltol'] = solvers.options['feastol'] = 1e-12

JS = r"""
const pr = Solver.prep(caso.X, caso.y, caso.std), b = Solver.solve(pr.G, pr.c, caso.lam, caso.alpha);
if (!b) return { nada: true };
const raw = Array.from(b, (v, j) => v/pr.sx[j]); let b0 = pr.my; raw.forEach((v, j) => { b0 -= pr.mx[j]*v; });
return { b: Array.from(b), raw, b0, sing: Solver.singular(pr.G) };
"""


def padroniza(X, std):
    X = np.asarray(X, float)
    mx = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd > 1e-12 * (1 + np.abs(mx)), sd, 1.0) if std else np.ones(X.shape[1])
    return (X - mx) / sd, mx, sd


def objetivo(Xs, y, b, lam, a):
    """RSS + λ[α‖β‖₁ + (1 − α)‖β‖²] com o intercepto ótimo (y e Xs centrados)."""
    r = (y - y.mean()) - (Xs - Xs.mean(0)) @ b
    return float(r @ r + lam * (a * np.abs(b).sum() + (1 - a) * (b @ b)))


def qp(Xs, y, lam, a):
    """Otimizador genérico: β = u − v com u, v ≥ 0; minimiza βᵀ(G + λ(1−α)I)β − 2cᵀβ + λα Σ(u + v)."""
    Xc, yc = Xs - Xs.mean(0), y - y.mean()
    p = Xs.shape[1]
    G = Xc.T @ Xc + lam * (1 - a) * np.eye(p)
    c = Xc.T @ yc
    M = np.block([[G, -G], [-G, G]])
    P = matrix(2 * M + 1e-12 * np.eye(2 * p))
    q = matrix(np.r_[-2 * c + lam * a, 2 * c + lam * a])
    s = solvers.qp(P, q, matrix(-np.eye(2 * p)), matrix(np.zeros(2 * p)))
    z = np.array(s['x']).ravel()
    return z[:p] - z[p:], s['status']


def sk(Xs, y, lam, a):
    n = len(y)
    if a == 0:
        m = Ridge(alpha=lam).fit(Xs, y)
    elif a == 1:
        m = Lasso(alpha=lam / (2 * n), max_iter=1_000_000, tol=1e-14).fit(Xs, y)
    else:
        m = ElasticNet(alpha=lam * (2 - a) / (2 * n), l1_ratio=a / (2 - a), max_iter=1_000_000, tol=1e-14).fit(Xs, y)
    return m.coef_, m.intercept_


def r_glmnet(casos):
    """Coeficientes do glmnet com a conversão mostrada pelo app (só casos com duas variáveis: o glmnet exige duas colunas)."""
    with tempfile.TemporaryDirectory(dir=os.environ.get('TEMP')) as d:
        linhas = []
        for k, c in enumerate(casos):
            Xs, _, _ = padroniza(c['X'], c['std'])
            n, a, lam = len(c['y']), c['alpha'], c['lam']
            # O glmnet padroniza y por dentro (desvio com divisor n), o que divide o termo ridge por esse desvio: a conversão compensa
            sy = float(np.std(c['y']))
            a1, a2 = lam * a / (2 * n), lam * (1 - a) / n
            gl = a1 + sy * a2
            ga = a1 / gl if gl > 0 else 1.0
            for i in range(n):
                linhas.append(','.join(map(str, [k, ga, gl, float(Xs[i, 0]), float(Xs[i, 1]), float(c['y'][i])])))
        arq = os.path.join(d, 'g.csv')
        with open(arq, 'w') as f:
            f.write('\n'.join(linhas) + '\n')
        scr = r"""
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); suppressPackageStartupMessages(library(glmnet))
A <- read.csv(commandArgs(TRUE)[1], header = FALSE)
for (k in unique(A$V1)) { B <- A[A$V1 == k, ]; X <- as.matrix(B[, 4:5]); y <- B$V6
  m <- tryCatch(glmnet(X, y, alpha = B$V2[1], lambda = B$V3[1], standardize = FALSE, control = list(thresh = 1e-14)), error = function(e) NULL)
  if (is.null(m)) cat(k, "ERRO\n") else cat(k, sprintf("%.17g", as.numeric(coef(m))), "\n") }
"""
        rf = os.path.join(d, 'g.R')
        with open(rf, 'w') as f:
            f.write(scr)
        r = subprocess.run([RS, rf, arq], capture_output=True, text=True)
    res = {}
    for ln in r.stdout.strip().splitlines():
        v = ln.split()
        res[int(v[0])] = None if v[1] == 'ERRO' else np.array([float(t) for t in v[1:]])
    return res, r.stderr[-500:]


def main():
    rng = np.random.default_rng(7)
    casos = gera()
    for i, c in enumerate(casos):
        c['alpha'] = [0.0, 1.0, 0.25, 0.5, 0.75, 1.0][(i // 2) % 6]
        c['lam'] = float(10 ** rng.uniform(-2, 4))
        c['std'] = i % 4 != 3
    out, ms = roda_js('penalizada', JS, casos)
    F = {k: [] for k in ['js', 'kkt_obj', 'sk', 'qp', 'glmnet']}
    M = {k: 0.0 for k in F}
    unicos = 0
    sel = []
    for i, (c, o) in enumerate(zip(casos, out)):
        if 'erro' in o or o.get('nada'):
            F['js'].append(i)
            continue
        y = np.array(c['y'])
        Xs, mx, sd = padroniza(c['X'], c['std'])
        b = np.array(o['b'])
        lam, a = c['lam'], c['alpha']
        oa = objetivo(Xs, y, b, lam, a)
        bq, st = qp(Xs, y, lam, a)
        oq = objetivo(Xs, y, bq, lam, a)
        esc = 1 + abs(oq)
        e = (oa - oq) / esc            # positivo: o app ficou pior que o otimizador genérico
        M['kkt_obj'] = max(M['kkt_obj'], e)
        if e > 1e-9: F['kkt_obj'].append((i, e, st))
        unico = not o['sing'] or a < 1
        unicos += unico
        if unico:
            e = float(np.max(np.abs(b - bq))) / (1 + float(np.max(np.abs(bq))))
            M['qp'] = max(M['qp'], e)
            if e > 1e-6: F['qp'].append((i, e))
            cs, ics = sk(Xs, y, lam, a)
            e = float(np.max(np.abs(np.r_[b, y.mean() - Xs.mean(0) @ b] - np.r_[cs, ics]))) / (1 + float(np.max(np.abs(cs))))
            M['sk'] = max(M['sk'], e)
            if e > 1e-6: F['sk'].append((i, e, a))
            if len(c['X'][0]) == 2:
                sel.append(i)
    G, err = r_glmnet([casos[i] for i in sel])
    nerr = 0
    erros_r = []
    for k, i in enumerate(sel):
        g = G.get(k)
        if g is None:
            nerr += 1
            yv = np.array(casos[i]['y'])
            erros_r.append((i, casos[i]['tipo'], 'y constante' if np.ptp(yv) == 0 else f'n = {len(yv)}'))
            continue
        b = np.array(out[i]['b'])
        y = np.array(casos[i]['y'])
        Xs, _, _ = padroniza(casos[i]['X'], casos[i]['std'])
        e = float(np.max(np.abs(np.r_[y.mean() - Xs.mean(0) @ b, b] - g))) / (1 + float(np.max(np.abs(b))))
        M['glmnet'] = max(M['glmnet'], e)
        if e > 1e-5: F['glmnet'].append((i, round(e, 4), casos[i]['alpha']))
    por_alpha = {}
    for i, e, a in F['glmnet']:
        por_alpha[a] = por_alpha.get(a, 0) + 1
    linhas = [f'penalizada.html: solver exato (padrões de sinal + KKT) em {len(casos)} conjuntos ({ms:.0f} ms no Node); λ log-uniforme em [0,01; 10⁴], ridge, lasso e elastic net (α = 0,25; 0,5; 0,75), com e sem padronizar',
              f'- sem solução no JS (ou erro): {len(F["js"])}',
              f'- função objetivo do app maior que a do otimizador genérico (QP do cvxopt), relativa > 1e-9: {len(F["kkt_obj"])}; maior diferença {M["kkt_obj"]:.2e}' + (f'; exemplos {F["kkt_obj"][:4]}' if F['kkt_obj'] else ''),
              f'- casos com solução única (XᵀX não singular, ou α < 1): {unicos}',
              f'- coeficientes contra o QP do cvxopt (casos únicos): {len(F["qp"])} divergências > 1e-6; maior {M["qp"]:.2e}' + (f'; exemplos {F["qp"][:4]}' if F['qp'] else ''),
              f'- coeficientes e intercepto contra Ridge/Lasso/ElasticNet do scikit-learn (tol = 1e-14): {len(F["sk"])} divergências > 1e-6; maior {M["sk"]:.2e}' + (f'; exemplos {F["sk"][:4]}' if F['sk'] else ''),
              f'- contra o glmnet ({len(sel)} casos únicos com duas variáveis, conversão mostrada na tela): {len(F["glmnet"])} divergências > 1e-5, por α: {por_alpha}; maior {M["glmnet"]:.2e}; erros no R: {nerr} {erros_r}' + (f'; exemplos {F["glmnet"][:6]}' if F['glmnet'] else '')]
    if err.strip() and nerr:
        linhas.append('stderr do R: ' + err)
    relatorio('t1_penalizada', linhas)


if __name__ == '__main__':
    main()
