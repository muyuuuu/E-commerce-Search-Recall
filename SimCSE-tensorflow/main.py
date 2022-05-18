import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_KERAS"] = '1'
from bert4keras.snippets import sequence_padding
from simcse_tf2.simcse import simcse
from simcse_tf2.data import get_tokenizer, load_data, SimCseDataGenerator
from simcse_tf2.losses import simcse_loss
import tensorflow as tf
import numpy as np


def texts_to_ids(data, tokenizer, max_len=128):
    """转换文本数据为id形式
    """
    token_ids = []
    for d in data:
        token_ids.append(tokenizer.encode(d, maxlen=128)[0])
    return sequence_padding(token_ids, length=128)


def encode_fun(texts, model, tokenizer, maxlen):
    inputs = texts_to_ids(texts, tokenizer, max_len=128)

    embeddings = model.predict([inputs, np.zeros_like(inputs)])
    return inputs, embeddings


class AdamWeightDecay(tf.keras.optimizers.Adam):
  """Adam enables L2 weight decay and clip_by_global_norm on gradients.
  Just adding the square of the weights to the loss function is *not* the
  correct way of using L2 regularization/weight decay with Adam, since that will
  interact with the m and v parameters in strange ways.
  Instead we want ot decay the weights in a manner that doesn't interact with
  the m/v parameters. This is equivalent to adding the square of the weights to
  the loss with plain (non-momentum) SGD.
  """

  def __init__(self,
               learning_rate=0.001,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-7,
               amsgrad=False,
               weight_decay_rate=0.0,
               include_in_weight_decay=None,
               exclude_from_weight_decay=None,
               gradient_clip_norm=1.0,
               name='AdamWeightDecay',
               **kwargs):
    super(AdamWeightDecay, self).__init__(learning_rate, beta_1, beta_2,
                                          epsilon, amsgrad, name, **kwargs)
    self.weight_decay_rate = weight_decay_rate
    self.gradient_clip_norm = gradient_clip_norm
    self._include_in_weight_decay = include_in_weight_decay
    self._exclude_from_weight_decay = exclude_from_weight_decay

  @classmethod
  def from_config(cls, config):
    """Creates an optimizer from its config with WarmUp custom object."""
    custom_objects = {'WarmUp': WarmUp}
    return super(AdamWeightDecay, cls).from_config(
        config, custom_objects=custom_objects)

  def _prepare_local(self, var_device, var_dtype, apply_state):
    super(AdamWeightDecay, self)._prepare_local(var_device, var_dtype,
                                                apply_state)
    apply_state[(var_device, var_dtype)]['weight_decay_rate'] = tf.constant(
        self.weight_decay_rate, name='adam_weight_decay_rate')

  def _decay_weights_op(self, var, learning_rate, apply_state):
    do_decay = self._do_use_weight_decay(var.name)
    if do_decay:
      return var.assign_sub(
          learning_rate * var *
          apply_state[(var.device, var.dtype.base_dtype)]['weight_decay_rate'],
          use_locking=self._use_locking)
    return tf.no_op()

  def apply_gradients(self,
                      grads_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):
    grads, tvars = list(zip(*grads_and_vars))
    if experimental_aggregate_gradients and self.gradient_clip_norm > 0.0:
      # when experimental_aggregate_gradients = False, apply_gradients() no
      # longer implicitly allreduce gradients, users manually allreduce gradient
      # and passed the allreduced grads_and_vars. For now, the
      # clip_by_global_norm will be moved to before the explicit allreduce to
      # keep the math the same as TF 1 and pre TF 2.2 implementation.
      (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    return super(AdamWeightDecay, self).apply_gradients(
        zip(grads, tvars),
        name=name,
        experimental_aggregate_gradients=experimental_aggregate_gradients)

  def _get_lr(self, var_device, var_dtype, apply_state):
    """Retrieves the learning rate with the given state."""
    if apply_state is None:
      return self._decayed_lr_t[var_dtype], {}

    apply_state = apply_state or {}
    coefficients = apply_state.get((var_device, var_dtype))
    if coefficients is None:
      coefficients = self._fallback_apply_state(var_device, var_dtype)
      apply_state[(var_device, var_dtype)] = coefficients

    return coefficients['lr_t'], dict(apply_state=apply_state)

  def _resource_apply_dense(self, grad, var, apply_state=None):
    # As the weight decay doesn't take any tensors from forward pass as inputs,
    # add a control dependency here to make sure it happens strictly in the
    # backward pass.
    # TODO(b/171088214): Remove it after the control dependency in
    # nested function is fixed.
    with tf.control_dependencies([tf.identity(grad)]):
      lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
      decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      return super(AdamWeightDecay,
                   self)._resource_apply_dense(grad, var, **kwargs)

  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    # As the weight decay doesn't take any tensors from forward pass as inputs,
    # add a control dependency here to make sure it happens strictly in the
    # backward pass.
    # TODO(b/171088214): Remove it after the control dependency in
    # nested function is fixed.
    with tf.control_dependencies([tf.identity(grad)]):
      lr_t, kwargs = self._get_lr(var.device, var.dtype.base_dtype, apply_state)
      decay = self._decay_weights_op(var, lr_t, apply_state)
    with tf.control_dependencies([decay]):
      return super(AdamWeightDecay,
                   self)._resource_apply_sparse(grad, var, indices, **kwargs)

  def get_config(self):
    config = super(AdamWeightDecay, self).get_config()
    config.update({
        'weight_decay_rate': self.weight_decay_rate,
    })
    return config

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if self.weight_decay_rate == 0:
      return False

    if self._include_in_weight_decay:
      for r in self._include_in_weight_decay:
        if re.search(r, param_name) is not None:
          return True

    if self._exclude_from_weight_decay:
      for r in self._exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applies a warmup schedule on a given learning rate decay schedule."""

  def __init__(self,
               initial_learning_rate,
               decay_schedule_fn,
               warmup_steps,
               power=1.0,
               name=None):
    super(WarmUp, self).__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.power = power
    self.decay_schedule_fn = decay_schedule_fn
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or 'WarmUp') as name:
      # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = (
          self.initial_learning_rate *
          tf.math.pow(warmup_percent_done, self.power))
      return tf.cond(
          global_step_float < warmup_steps_float,
          lambda: warmup_learning_rate,
          lambda: self.decay_schedule_fn(step),
          name=name)

  def get_config(self):
    return {
        'initial_learning_rate': self.initial_learning_rate,
        'decay_schedule_fn': self.decay_schedule_fn,
        'warmup_steps': self.warmup_steps,
        'power': self.power,
        'name': self.name
    }


if __name__ == '__main__':
    # 1. bert config /home/20031211375/15-epoch
    model_path = '/home/20031211375/15-epoch/random100'
    checkpoint_path = '%s/bert_model.ckpt' % model_path
    config_path = '%s/bert_config.json' % model_path
    dict_path = '%s/vocab.txt' % model_path

    # 2. set hyper parameters
    max_len = 115
    dropout_rate = 0.1
    batch_size = 60
    learning_rate = 3e-5
    epochs = 15
    output_units = 128
    activation = 'tanh'
    save_path = "training_1/model.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    # 3. data generator
    train_data = load_data('./query_doc.csv', delimiter="\t")
    train_data_size = len(train_data)
    steps_per_epoch = int(train_data_size / batch_size)
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = 0.1 * num_train_steps
    decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
          initial_learning_rate=3e-5,
          decay_steps=num_train_steps,
          end_learning_rate=0)
    warmup_schedule = WarmUp(
            initial_learning_rate=learning_rate,
            decay_schedule_fn=decay_schedule,
            warmup_steps=warmup_steps)
    optimizer = AdamWeightDecay(
            learning_rate=warmup_schedule,
            weight_decay_rate=0.01,
            epsilon=1e-6,
            exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])
    # train_data = load_data('./examples/data/sup_sample.csv', delimiter = "\t")
    train_generator = SimCseDataGenerator(train_data, dict_path, batch_size, max_len)
    print(next(train_generator.forfit()))

    # 4. build model
    model = simcse(config_path, checkpoint_path, dropout_rate=dropout_rate, output_units=output_units,
                   output_activation=activation)
    print(model.summary())
    # 5. model compile
    model.compile(loss=simcse_loss, optimizer=optimizer)

    # 6. model fit
    model.fit(train_generator.forfit(), steps_per_epoch=len(train_generator), epochs=epochs, callbacks=[cp_callback])

    model.load_weights(save_path)

    import csv
    from tqdm import tqdm

    pre_batch_size = 4000
    corpus = [line[1] for line in csv.reader(open("./data/corpus_clean.txt"), delimiter='\t')]
    query = [line[1] for line in csv.reader(open("./data/dev.query_clean.txt"), delimiter='\t')]
    tokenizer = get_tokenizer(dict_path)
    query_embedding_file = csv.writer(open('./query_embedding', 'w'), delimiter='\t')

    for i in tqdm(range(0, len(query), pre_batch_size)):
        batch_text = query[i:i + pre_batch_size]
        print("query size:", len(batch_text))
        ids, temp_embedding = encode_fun(batch_text, model, tokenizer, maxlen=128)
        for j in range(len(temp_embedding)):
            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            ids_str = ids[j].tolist()
            ids_str = [str(t) for t in ids_str]
            ids_str = ','.join(ids_str)
            str_ = str(i+j+200001) + "\t" + writer_str + "\t" + ids_str + '\n'
            with open('./query_embedding', 'a+') as f:
                f.write(str_)
    print("query end!")
    doc_embedding_file = csv.writer(open('./doc_embedding', 'w'), delimiter='\t')
    for i in tqdm(range(0, len(corpus), pre_batch_size)):
        batch_text = corpus[i:i + pre_batch_size]
        ids, temp_embedding = encode_fun(batch_text, model, tokenizer, maxlen=128)
        for j in range(len(temp_embedding)):
            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            ids_str = ids[j].tolist()[1:] + [0]
            ids_str = [str(t) for t in ids_str]
            ids_str = ','.join(ids_str)
            str_ = str(i+j+1) + "\t" + writer_str + "\t" + ids_str + '\n'
            with open('./doc_embedding', 'a+') as f:
                f.write(str_)
    print("doc end!")
