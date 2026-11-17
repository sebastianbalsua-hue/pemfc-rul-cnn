import tensorflow as tf

def build_model(input_shape, dropout=0.25):
    L=tf.keras.layers
    return tf.keras.Sequential([L.Input(shape=input_shape),
        L.Conv1D(32,3,activation='relu'),L.MaxPooling1D(2),
        L.Conv1D(64,3,activation='relu'),L.MaxPooling1D(2),
        L.LSTM(64),L.Dense(32,activation='relu'),L.Dense(1)],name='cnn_lstm')
