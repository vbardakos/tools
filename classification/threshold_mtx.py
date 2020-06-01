"""
TF Confusion Matrix with a given threshold
@ author vbar
"""
import tensorflow as tf

def threshold_matrix(labels, probs, threshold=0):
    """
    tf.math.confusion_matrix with minimum threshold
    
    Params:
        labels (Tensor): real labels
        probs (Tensor) : P(label)
        threshold (float) : gives output for P in (threshold,1]

    Label P(x) : [[.6,.4],[.7,.3],[.8,.2],[.9,.1]]
    Actual Val : [0,1,0,1] # Predictions : True, False, True, False

    > threshold_matrix(y_pred, y_true, 0.7)
    [[1, 0], - True positive
    [2, 0]] - False negative
    """
    assert 0 <= threshold < 1
    t_map = lambda x : 1 if tf.math.reduce_max(x) > threshold else 0
    thres = tf.map_fn(t_map, probs)

    pred  = tf.boolean_mask(tf.argmax(probs,1),thres)
    labs  = tf.boolean_mask(tf.cast(labels, tf.int64),thres)

    return tf.math.confusion_matrix(labs,pred)