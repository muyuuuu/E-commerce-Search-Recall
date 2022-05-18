# -*- coding:utf-8 -*-
"""
Author:
    jifei, jifei@outlook.com
"""

import tensorflow as tf


def simcse_loss(y_true, y_pred):
    """
    simcse loss
    """
    idxs = tf.range(0, tf.shape(y_pred)[0])
    idxs_1 = idxs[None, :]
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]
    y_true = tf.equal(idxs_1, idxs_2)
    y_true = tf.cast(y_true, tf.keras.backend.floatx())
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    similarities = tf.matmul(y_pred, y_pred, transpose_b=True)
    similarities = similarities - tf.eye(tf.shape(y_pred)[0]) * 1e12
    similarities = similarities / 0.05
    loss = tf.keras.losses.categorical_crossentropy(y_true, similarities, from_logits=True)
    return tf.reduce_mean(loss)


def simcse_hard_neg_loss(y_true, y_pred):
    """
    simcse loss for hard neg or random neg
    """
    row = tf.range(0, tf.shape(y_pred)[0], 3)
    col = tf.range(tf.shape(y_pred)[0])
    col = tf.squeeze(tf.where(col % 3 != 0), axis=1)
    y_true = tf.range(0, len(col), 2)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    similarities = tf.matmul(y_pred, y_pred, transpose_b=True)
    similarities = tf.gather(similarities, row, axis=0)
    similarities = tf.gather(similarities, col, axis=1)
    similarities = similarities / 0.05
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, similarities, from_logits=True)
    return tf.reduce_mean(loss)
