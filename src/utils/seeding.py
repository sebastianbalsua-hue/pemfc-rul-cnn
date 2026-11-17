import random
import numpy as np

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
