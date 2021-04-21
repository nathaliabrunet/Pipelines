#### PIPELINE COM SCIKIT-LEARN ###
# https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


# Cria um dataset para usar no exemplo
X, y = datasets.make_regression(n_samples=3000, 
                                n_features=15, 
                                random_state=42)
X = pd.DataFrame(X)
X['target'] = X[0].apply(lambda x: 'True' if x > 1 else 'False')
X = X.values

# Separa o conjunto em treino e teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


"""
PASSOS
Passo 1: OneHotEncoding na coluna 'Target'
Passo 2: Imputando dados faltantes.
Passo 3: Aplicando função para transformação dos valores das colunas (transformação vai depender do problema, claro :))
Passo 4: Normalização dos dados
Passo 5: PCA - Redução de dimensionalidade.
Passo 6: Aplicando RandomForest para regressão
Passo 7 (só no teste): Predict utilizando o modelo do PASSO 6 no conjunto de teste.
"""


pipe = Pipeline(steps =[
    ('passo1_one_hot_encoding', ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [15]),],remainder='passthrough')),
    # ColumnTransformer:
    # name = 'one_hot_encoder'
    # ação = OneHotEncoder()
    # coluna que vai ser aplicada a ação = [15] #índice 15 (onde tem o target)

    ('passo2_imputer', SimpleImputer(strategy='mean')),
    ('passo3_transform_func', FunctionTransformer(lambda x: x * 2, validate=True)),
    ('passo4_scaler', MinMaxScaler(feature_range=(0, 1))),
    ('passo5_pca', PCA(n_components=5, random_state=42)),
    ('passo6_model', RandomForestRegressor(n_estimators=100, random_state=42)) #n_estimators=100, 
])

print('pipe steps:\n', pipe.steps)
#pipe.steps
# Fit usando pipe (passa por todos os passos setados no pipe)
pipe.fit(X_train, y_train)

# Predict usando o pipe
y_pred_pipe = pipe.predict(X_test)

#visualizando 5 predicts
y_pred_pipe[:5]


# Cross-Validation e Grid-Search com Pipelines
# Tunando hiperparâmetros com 5-fold cross-validation e pipelines
# tem que ter dois sublinhados entre o nome do estimador e os parâmetros em um Pipeline
parameters = {'passo6_model__n_estimators': [100, 150], 
              'passo6_model__max_features': ['log2', 'sqrt','auto'],
              'passo6_model__max_depth': [3, 5], 
              'passo6_model__min_samples_split': [2, 3],
              'passo6_model__min_samples_leaf': [5,8],
              'passo6_model__bootstrap': [True, False]}

metricas = ["explained_variance",
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
            #  "neg_mean_squared_log_error",
            "neg_median_absolute_error",
            "r2"]

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(pipe, #model vai ser o pipe
                    param_grid=parameters,
                    scoring=metricas,
                    cv=kfold, 
                    n_jobs=-1,
                    #verbose=1, #serve p/ escolher o nível de mensagens que nossa grid vai emitir enquanto treina e testa todas as combinações de parâmetros.
                    
                    refit="neg_mean_squared_error" #define métrica "favorita"
                    # vão ser testadas várias combinações de parâmetros, No refit apontamos a métrica "favorita" 
                    #e vai fazer com que a grid já devolva o modelo que foi melhor naquela métrica
                    
                    #return_train_score=False  #não mostra, porque só queremos saber o resultado do teste
                   )

grid.fit(X_train, y_train)

# melhor parâmetro
print('\nbest_params_:\n', grid.best_params_ )


pd.set_option("max_columns",200) #para enxergar mais colunas
#to dataframe
results = pd.DataFrame(grid.cv_results_)
#ordenando pelo melhores valores
#results.sort_values(by="rank_test_neg_median_absolute_error").head()