import importlib
import tensorflow as tf
from src.training.callbacks import callbacks
from src.utils.seeding import seed_everything

MODELS={'cnn':'cnn_causal_dilated','cnn_lstm':'cnn_lstm_baseline','lstm':'lstm_baseline','gru':'gru_baseline'}

def train(x,y,xval,yval,kind='cnn',epochs=100,batch_size=32,seed=42,optimizer='adam',learning_rate=0.001,patience=10):
    seed_everything(seed)
    model=importlib.import_module('src.models.'+MODELS[kind]).build_model(x.shape[1:])
    if optimizer not in ('adam','sgd'): raise ValueError('Optimizador debe ser adam o sgd')
    opt=tf.keras.optimizers.Adam(learning_rate) if optimizer=='adam' else tf.keras.optimizers.SGD(learning_rate,momentum=0.9)
    model.compile(optimizer=opt,loss='mse')
    history=model.fit(x,y,validation_data=(xval,yval),epochs=epochs,batch_size=batch_size,
                      callbacks=callbacks(patience),shuffle=True,verbose=2)
    return model,history.history
