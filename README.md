#  PIPELINE COM SCIKIT-LEARN

## Tópicos 

[Descrição](#Descrição)

[Passos seguidos](#Passos-seguidos)

[Ferramentas utilizadas](#Ferramentas-utilizadas)

## Descrição

Diretório só para brincar com o uso de pipeline com scikit-learn, aplicando Cross-Validation e Grid-Search
https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

## Passos seguidos

```
PASSOS
Passo 1: OneHotEncoding na coluna 'Target'
Passo 2: Imputando dados faltantes.
Passo 3: Aplicando função para transformação dos valores das colunas (transformação vai depender do problema, claro :))
Passo 4: Normalização dos dados
Passo 5: PCA - Redução de dimensionalidade.
Passo 6: Aplicando RandomForest para regressão
Passo 7 (só no teste): Predict utilizando o modelo do PASSO 6 no conjunto de teste.
```

## Ferramentas utilizadas
* Jupyter notebook
* Python