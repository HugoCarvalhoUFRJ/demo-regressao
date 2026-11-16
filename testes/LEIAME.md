# Testes

Os scripts ficam aqui. Tudo o que eles instalam ou gravam fica em `C:\Users\hugom\AppData\Local\classificadores-testes` (a mesma pasta dos testes dos classificadores), fora do OneDrive; os relatórios e capturas vão para `resultados-regressao/`.

- `venv/`: Python com `--system-site-packages` a partir do conda base (scikit-learn 1.7.2, statsmodels 0.14.5), mais `playwright`, `cvxopt` e `axe-playwright-python`.
- `Rlib/`: pacotes de R usados aqui, além dos dos classificadores: `glmnet`, `FNN`, `gbm` (o `rpart` é o do próprio R).
- O Node que roda os solvers é o do pacote `playwright`; o navegador é o Chrome instalado, com perfil temporário.

## Como rodar (Git Bash)

```bash
cd testes
source env.sh                 # aponta TEMP, caches e bibliotecas para a pasta de testes
"$PY" t1_linear.py            # mínimos quadrados: 1000 conjuntos (scikit-learn, statsmodels, lm; mpmath como árbitro)
"$PY" t1_penalizada.py        # ridge, lasso, elastic net: 1000 conjuntos (scikit-learn, QP do cvxopt, glmnet)
"$PY" t1_penalizada_poli.py   # idem com polinômio de grau 2–10 (conjunto ativo): 1000 conjuntos de uma variável
"$PY" t_viesvar.py            # gráfico de viés e variância e Variabilidade contra a conta refeita com scikit-learn/numpy
"$PY" t1_knn.py               # KNN: 1000 conjuntos (KNeighborsRegressor, FNN::knn.reg, kknn)
"$PY" t1_arvores.py todos     # ou arvore, boosting, floresta (DecisionTreeRegressor, rpart, GradientBoostingRegressor; floresta em distribuição)
"$PY" t_casca.py              # a casca dos seis apps e o núcleo das árvores são idênticos? ("aplicar" copia a casca do knn.html)
"$PY" t_nav.py todos          # ou e2e, aleatorio, axe, contextos, desempenho, capturas
"$PY" t_rapido.py knn         # verificação rápida durante o desenvolvimento (--cel: tela de celular)
```

- `dados.py` gera os 1000 conjuntos (lineares, não lineares, aleatórios, em grade com repetições, casos extremos), metade com uma variável e metade com duas.
- `comum.py` extrai o bloco `/*SOLVER-START*/ … /*SOLVER-END*/` de cada app e o executa no Node; `base.py` tem o servidor local e as utilidades do navegador.
- `t_nav.py e2e` lê o estado da tela pelo gancho `window.__regr` e executa literalmente as linhas de R e de Python mostradas.
