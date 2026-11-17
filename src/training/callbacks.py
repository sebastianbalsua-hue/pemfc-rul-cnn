import tensorflow as tf

def callbacks(patience=10):
    return [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience,restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=max(1,patience//2))]
