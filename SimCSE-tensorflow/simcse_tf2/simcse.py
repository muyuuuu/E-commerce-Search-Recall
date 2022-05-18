# -*- coding:utf-8 -*-
"""
Author:
    jifei, jifei@outlook.com
"""
import tensorflow as tf
import os
import json

os.environ["TF_KERAS"] = '1'
from bert4keras.models import build_transformer_model


def simcse(config_path,
           checkpoint_path,
           model='roformer',
           pooling='first-last-avg',
           dropout_rate=0.2,
           output_units=None,
           output_activation=None,
           ):
    """Build SimCSE model

    :param config_path:string
    :param checkpoint_path:string
    :param model:string, model name
    :param pooling:string, in ['first-last-avg', 'last-avg', 'cls', 'pooler']
    :param dropout_rate:float
    :param output_units:int
    :param output_activation:string
    :return: A Keras model instance.
    """
    assert pooling in ['first-last-avg', 'last-avg', 'cls', 'pooler']
    with open(config_path, 'r') as load_f:
        num_hidden_layers = json.load(load_f)['num_hidden_layers']

    bert = build_transformer_model(
        config_path,
        checkpoint_path,
        model='roformer',
        # application='unilm',
        with_pool='linear',
        # with_mlm='linear',
        dropout_rate=0.2,
        ignore_invalid_weights=True
    )

    last_layer_output = bert.get_layer('Transformer-%d-FeedForward-Norm' % (num_hidden_layers - 1)).output
    if pooling == 'first-last-avg':
        outputs = [
            tf.keras.layers.GlobalAveragePooling1D()(bert.get_layer('Transformer-%d-FeedForward-Norm' % 0).output),
            tf.keras.layers.GlobalAveragePooling1D()(last_layer_output)
        ]
        output = tf.keras.layers.Average()(outputs)
    elif pooling == 'last-avg':
        output = tf.keras.layers.GlobalAveragePooling1D()(last_layer_output)
    elif pooling == 'cls':
        output = tf.keras.layers.Lambda(lambda x: x[:, 0])(last_layer_output)
    else:
        output = bert.output

    if output_units and output_activation:
        output = tf.keras.layers.Dense(output_units, activation=output_activation)(output)
    model = tf.keras.Model(bert.inputs, output)
    return model
