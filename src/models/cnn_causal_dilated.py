"""Principal architecture from the supplied technical document, section 7.1."""
import tensorflow as tf

def build_model(input_shape, dropout=0.25):
    L=tf.keras.layers
    inputs=L.Input(shape=input_shape); x=inputs
    for i,(channels,dilation) in enumerate(zip((32,64,64,128),(1,2,4,8))):
        x=L.Conv1D(channels,3,dilation_rate=dilation,padding='causal',activation='relu')(x)
        x=L.BatchNormalization()(x)
        if i==0: x=L.MaxPooling1D(2)(x)
    x=L.GlobalAveragePooling1D()(x)
    x=L.Dense(64,activation='relu')(x); x=L.Dropout(dropout)(x)
    x=L.Dense(32,activation='relu')(x); x=L.Dropout(0.2)(x)
    return tf.keras.Model(inputs,L.Dense(1)(x),name='cnn_causal_dilated')
