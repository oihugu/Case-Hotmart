import keras
import tensorflow as tf
from keras import layers
import keras_tuner as kt

#Import sklearn clasification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC




def m_build_model(input_shape):
    def build_model(hp):
        inputs = keras.Input(shape=(input_shape))
        x = layers.Flatten()(inputs) 
        x = layers.Dense(units=hp.Choice('dense_units', [32,64,128]), activation="relu")(x)
        x = layers.Dense(units=hp.Choice('dense_units_2', [8,16,32]), activation=hp.Choice('activation', ['relu', 'sigmoid', 'tanh']))(x)
        outputs = layers.Dense(2, activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'], default='adam'), 
                      loss=hp.Choice('loss', ['binary_crossentropy', 'categorical_crossentropy']),
                      metrics=['accuracy', 'AUC', 'Precision', 'Recall'])        
        return model
    return build_model

def build_model_sklearn(hp):
    model_type = hp.Choice('model_type', ['LR', 'RF', 'SVM'])
    if model_type == 'LR':
        model = LogisticRegression(C=hp.Float('C', min_value=0.1, max_value=1.0, sampling='log'))
    elif model_type == 'RF':
        model = RandomForestClassifier(max_depth=hp.Int('max_depth', min_value=3, max_value=10, step=1),
                                       n_estimators=hp.Int('n_estimators', min_value=10, max_value=50, step=10))
    else:
        model = SVC(C=hp.Float('C', min_value=0.1, max_value=1.0, sampling='log'))
    return model
    