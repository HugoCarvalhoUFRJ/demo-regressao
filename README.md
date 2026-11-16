# Regressão Interativa

Apps para aprender métodos de regressão arrastando pontos num gráfico. A cada movimento o modelo é recalculado e o ajuste, os erros e os números na tela se atualizam na hora.

**Acesse:** <https://hugocarvalho.mat.br/demo-regressao/>

Cada guia tem endereço próprio, útil para mandar à turma: por exemplo, [`#penalizada`](https://hugocarvalho.mat.br/demo-regressao/#penalizada) ou [`#boosting`](https://hugocarvalho.mat.br/demo-regressao/#boosting).

## Os apps

| Guia | O que mostra | Reproduz | Página |
|---|---|---|---|
| Mínimos quadrados | Reta, polinômio de grau 1 a 10 ou plano (com interação), com erros-padrão, t, p-valores, RSE, R² e F; resíduos e seus quadrados; as curvas de nível do RSS em (β₀, β₁), como na Fig. 3.2; alavancagem e resíduo studentizado; MSE de treino e de teste contra o grau, como na Fig. 2.9 | `LinearRegression`, `statsmodels.OLS`, `lm` | [`linear.html`](linear.html) |
| Ridge, lasso, EN | Ridge, lasso e elastic net com solução exata: o plano dos coeficientes com as elipses do RSS e a região de restrição (Fig. 6.7), o caminho dos coeficientes (Figs. 6.4 e 6.6) e, com uma variável, o encolhimento contra o soft-thresholding (Fig. 6.10). Com uma variável, também um polinômio de grau até 10, com os coeficientes de cada termo num gráfico de hastes, para ver a penalidade agindo em mais dimensões. Dá para arrastar β̂ e mudar a correlação entre X₁ e X₂ | `Ridge`, `Lasso`, `ElasticNet`, `glmnet` | [`penalizada.html`](penalizada.html) |
| KNN | Regressão por K vizinhos mais próximos, com pesos iguais ou 1/d e distância com p = 1, 2 ou 3; a vizinhança e a média sob o cursor | `KNeighborsRegressor`, `FNN::knn.reg`, `kknn` | [`knn.html`](knn.html) |
| Árvore | Árvore de regressão com partição exata (Fig. 8.3), diagrama com as médias das folhas e poda pela Eq. (8.4); passar o mouse num nó realça a região dele | `DecisionTreeRegressor`, `rpart` | [`arvore.html`](arvore.html) |
| Bagging e floresta | Bagging de árvores ou de regressões lineares e floresta aleatória, com os ajustes de cada amostra bootstrap e o erro OOB | `RandomForestRegressor`, `BaggingRegressor` | [`floresta.html`](floresta.html) |
| Boosting | O Algoritmo 8.2 passo a passo: os resíduos, a árvore b ajustada a eles e o MSE de treino e de teste a cada árvore, com animação | `GradientBoostingRegressor` | [`boosting.html`](boosting.html) |

A coluna "Reproduz" indica as implementações de referência (scikit-learn, statsmodels ou R) cujo resultado o app replica. O `randomForest` e o `gbm` do R também aparecem no código, mas dão resultados só parecidos, e a tela diz por quê.

## O que dá para fazer

- **Escolher uma ou duas variáveis.** Com uma variável, o gráfico é (X, Y) e cada ponto se arrasta em qualquer direção. Com duas, o gráfico é (X₁, X₂); o fundo mostra ŷ em faixas de cor, com as curvas de nível, e cada ponto tem a cor do seu y. Arrastar move o ponto em (X₁, X₂), e a ferramenta "Mudar y" (ou PageUp e PageDown) muda o y dele.
- **Arrastar, acrescentar e remover pontos** com mouse, toque ou teclado (setas movem o ponto focado, Shift dá passo maior, Delete remove). Durante o arraste com uma variável, o ajuste anterior aparece em cinza.
- **Escolher o conjunto de dados.** Cada um vem de uma f verdadeira conhecida mais um erro, yᵢ = f(xᵢ) + σ·zᵢ, com semente fixa: a turma inteira vê sempre os mesmos pontos. Com uma variável: reta, curva suave (como na Fig. 2.9), quadrática (como na Fig. 3.8), seno, degraus, reta com poucos pontos e um conjunto com outlier e ponto de alta alavancagem (como nas Figs. 3.12 e 3.13). Com duas: plano, só X₁ importa, X₁ e X₂ correlacionados, plano com interação, degraus, sela e chapéu.
- **Mudar o ruído σ** e **mostrar a f verdadeira**, tracejada.
- **Ligar um conjunto de teste**, outra amostra da mesma distribuição, e comparar o MSE de treino com o de teste e com o erro irredutível σ².
- **Ver a variabilidade** (uma variável): o mesmo método ajustado em 40 amostras novas (20 delas desenhadas), com o viés² e a variância estimados, como na Eq. (2.7).
- **Ver o balanço entre viés e variância**, como na Fig. 2.12: viés², variância e MSE de teste esperado contra o parâmetro de flexibilidade de cada método (grau, λ, 1/K, α da poda, número de árvores), calculados em 40 amostras novas da f verdadeira. Clicar no gráfico escolhe o parâmetro.
- **Girar o gráfico 3D** (duas variáveis): a superfície ajustada, com os pontos na altura do seu y, os resíduos e, se quiser, a f verdadeira. Nos métodos constantes por partes (KNN e árvores) a superfície tem degraus.
- **Copiar o código equivalente** em R e em Python, com todos os parâmetros explícitos. Quando a escala de um parâmetro muda entre o livro e a biblioteca (λ do lasso, `ccp_alpha`, o `glmnet`), a tela mostra a conversão.

O tema claro ou escuro é escolhido no cabeçalho e vale para todas as guias.

## Como funciona

O site é estático e não tem etapa de compilação. Cada app é um único arquivo HTML com CSS e JavaScript embutidos, e todo o cálculo roda no navegador, rápido o bastante para refazer o ajuste a cada quadro do arraste. A única dependência externa são as fontes do Google Fonts, porém os apps funcionam sem elas.

As soluções são exatas: mínimos quadrados por QR, ridge por fórmula fechada, lasso e elastic net testando cada padrão de sinais e conferindo as condições KKT (até duas colunas) ou por conjunto ativo (polinômio com mais colunas), e árvores, florestas e boosting desenhados pela grade dos limiares. A floresta usa uma semente fixa por amostra bootstrap, para que o arraste não sorteie tudo de novo; o botão "Sortear de novo" troca as sementes.

Para usar sem internet, baixe o repositório e abra o `index.html` no navegador. Cada app também funciona aberto sozinho.

## Verificação

A pasta [`testes/`](testes/) tem os scripts usados para conferir os apps (veja [`testes/LEIAME.md`](testes/LEIAME.md)):

- o solver de cada app é comparado, em 1000 conjuntos de dados aleatórios com variedade planejada, com o scikit-learn e o statsmodels e, quando possível, com o R (`lm`, `glmnet`, `FNN`, `kknn`, `rpart`) e com um otimizador genérico (cvxopt); a floresta é comparada em distribuição;
- com Playwright, testes ponta a ponta leem o estado da tela e executam literalmente as linhas de R e Python exibidas;
- centenas de interações aleatórias por app procuram `NaN`, erros no console e estouro de layout;
- axe-core confere a acessibilidade de todas as páginas, nos temas claro e escuro;
- tudo isso roda servido por HTTP e aberto direto do disco, dentro das guias e com cada app sozinho, no computador e numa tela de celular de 390 px com toque simulado.

## Estrutura do repositório

```
index.html        página principal com as 6 guias
linear.html       mínimos quadrados
penalizada.html   ridge, lasso e elastic net
knn.html          regressão KNN
arvore.html       árvore de regressão
floresta.html     bagging e floresta aleatória
boosting.html     boosting de árvores
testes/           scripts de verificação (Python, R e Playwright)
```

## Sobre

Estes apps foram desenvolvidos por motivação pedagógica, a partir de perguntas dos alunos em sala de aula, para visualizar melhor os métodos de regressão, usando IA: Claude, Opus 5.5 no esforço alto.

Disciplina Aprendizagem de Máquina I, da UFRJ, ministrada pelo Prof. Hugo Carvalho. Os materiais didáticos do curso estão em <https://github.com/HugoCarvalhoUFRJ/ap-maq>.

A notação é a mesma do livro [*An Introduction to Statistical Learning*](https://www.statlearning.com/). As Eqs. citadas nos apps usam a numeração do livro (edição com aplicações em Python).

Os conjuntos "como na Fig. 2.9", "como na Fig. 3.8", "como nas Figs. 3.12 e 3.13" e "como na Eq. (3.31)" são gerados com a mesma ideia do livro, não com os dados dele. A elastic net não está no texto do capítulo 6 (aparece no laboratório da seção 6.5); a forma usada aqui está descrita na própria guia.

## Licença

© 2026 Hugo Carvalho. Este trabalho está licenciado sob a [Creative Commons Atribuição-NãoComercial 4.0 Internacional (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/deed.pt-br). Você pode usar, copiar e adaptar os apps, inclusive em outras aulas, desde que cite a fonte e não os use para fins comerciais. O texto completo está em [`LICENSE`](LICENSE).
