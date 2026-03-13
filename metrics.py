import tensorflow as tf

def mse(y_true, y_pred):
    """Mean Squared Error."""
    return tf.reduce_mean(tf.square(y_true - y_pred))

def psnr(y_true, y_pred):
    """Peak Signal-to-Noise Ratio."""
    # Since outputs are AB channels normalized between [-1, 1], the range is 2.0
    return tf.image.psnr(y_true, y_pred, max_val=2.0)

def ssim(y_true, y_pred):
    """Structural Similarity Index."""
    # Values between [-1, 1]
    return tf.image.ssim(y_true, y_pred, max_val=2.0)
