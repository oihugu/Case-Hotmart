import keras
import tensorflow as tf
from keras import layers
import keras_tuner as kt

#Import sklearn regression models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

def m_build_model_regression(input_shape): #Função Geradora / Factory Function
    def build_model_regression(hp):
        model = tf.keras.Sequential([
            layers.Dense(hp.Choice('Dense 1', values=[16,32,64]), activation='relu', input_shape=(input_shape),
                        kernel_regularizer=tf.keras.regularizers.l2(hp.Choice('Dense 1 - l2', values=[0.1, 0.05, 0.01, 0.001, 0.0001]))),
            layers.Dense(hp.Choice('Dense 2', values=[8,16,32]), activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(hp.Choice('Dense 2 - l2', values=[0.1, 0.05, 0.01, 0.001, 0.0001]))),
            layers.Dropout(hp.Float('Dropout', min_value=0.0, max_value=0.5, default=0.25, step=0.1)),
            layers.Dense(2)
        ])
        optimizer = tf.keras.optimizers.RMSprop(hp.Choice('RMSprop', values=[0.001, 0.01, 0.1]))
        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=[keras.metrics.MeanSquaredLogarithmicError(name='msle'),
                               keras.metrics.MeanAbsoluteError(name='mae')]) # Usando MSLE pois não temos tanta precisão nos dados, mas assim podemos ter uma ideia 
        return model
    return build_model_regression



def build_model_sklearn_regression(hp):
    model_type = hp.Choice('model_type', ['LR', 'RF', 'SVR', 'Ridge', 'Lasso', 'ElasticNet'])
    ## Regressão Linear Simples
    if model_type == 'LR':
        model = MultiOutputRegressor(LinearRegression())
    ## Random Forest Regressor
    ## Não linear
    ## Fácil de ser explicado por ser mais trasnparente que os demais
    elif model_type == 'RF':
        model = MultiOutputRegressor(RandomForestRegressor(max_depth=hp.Int('max_depth', min_value=3, max_value=10, step=1),
                                       n_estimators=hp.Int('n_estimators', min_value=10, max_value=50, step=10)))
    # Support Vector Regressor
    # Modelo de regressão que utiliza vetores de suporte
    # Não linear
    # C: penalidade por erro
    # epsilon: margem de erro
    elif model_type == 'SVR':
        model = MultiOutputRegressor(SVR(C=hp.Float('C', min_value=1e-3, max_value=1e3, sampling='LOG', default=1),
                    epsilon=hp.Float('epsilon', min_value=1e-3, max_value=1e3, sampling='LOG', default=1)))
    #########################################
    ###### ElasticNet, Ridge, Lasso #########
    #########################################
    # Esses tipos de modelos já possuem regularização
    # Modelos lineares
    # assim diminuindo o overfitting e aumentando a generalização
    # alpha: penalidade por erro
    # l1_ratio: proporção entre a penalidade L1 e L2
    elif model_type == 'ElasticNet':
        model = MultiOutputRegressor(ElasticNet(alpha=hp.Float('alpha', min_value=1e-3, max_value=1, sampling='LOG', default=1),
                           l1_ratio=hp.Float('l1_ratio', min_value=1e-3, max_value=1, sampling='LOG', default=1)))
    elif model_type == 'Ridge':
        model = MultiOutputRegressor(Ridge(alpha=hp.Float('alpha', min_value=1e-3, max_value=1, sampling='LOG', default=1)))
    elif model_type == 'Lasso':
        model = MultiOutputRegressor(Lasso(alpha=hp.Float('alpha', min_value=1e-3, max_value=1, sampling='LOG', default=1)))
    return model
