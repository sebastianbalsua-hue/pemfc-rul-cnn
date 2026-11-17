import tensorflow as tf

def build_model(input_shape, dropout=0.25):
    L=tf.keras.layers
    return tf.keras.Sequential([L.Input(shape=input_shape),L.GRU(64,return_sequences=True),
        L.GRU(32),L.Dense(32,activation='relu'),L.Dense(1)],name='gru')
