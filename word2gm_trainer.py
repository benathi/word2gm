"""
Ben Athiwaratkun

Training code for Gaussian Mixture word embeddings model

Adapted from tensorflow's word2vec.py
(https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time
import math
# Retrict to CPU only
os.environ["CUDA_VISIBLE_DEVICES"]=""

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf

#from tensorflow.models.embedding import gen_word2vec as word2vec
#word2vec = tf.load_op_library(os.path.join(os.path.di))
word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model and "
                    "training summaries. (required)")
flags.DEFINE_string("train_data", None, "Training text file. (required)")
flags.DEFINE_integer("embedding_size", 50, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 5,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")

flags.DEFINE_integer("batch_size", 256,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
flags.DEFINE_integer("min_count", 5,
                     "The minimum number of word occurrences for it to be "
                     "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")
flags.DEFINE_integer("statistics_interval", 5,
                     "Print statistics every n seconds.")
flags.DEFINE_integer("summary_interval", 5,
                     "Save training summary to file every n seconds (rounded "
                     "up to statistics interval).")
flags.DEFINE_integer("checkpoint_interval", 600,
                     "Checkpoint the model (i.e. save the parameters) every n "
                     "seconds (rounded up to statistics interval).")
flags.DEFINE_integer("num_mixtures", 2,
                     "Number of mixture component for Mixture of Gaussians")
flags.DEFINE_boolean("spherical", False,
                     "Whether the model should be spherical of diagonal"
                     "The default is spherical")

flags.DEFINE_float("var_scale", 0.05, "Variance scale")

flags.DEFINE_boolean("ckpt_all", False, "Keep all checkpoints"
                      "(Warning: This requires a large amount of disk space).")

flags.DEFINE_float("norm_cap", 3.0,
                   "The upper bound of norm of mean vector")
flags.DEFINE_float("lower_sig", 0.02,
                   "The lower bound for sigma element-wise")
flags.DEFINE_float("upper_sig", 5.0,
                   "The upper bound for sigma element-wise")
flags.DEFINE_float("mu_scale", 1.0,
                  "The average norm will be around mu_scale")

flags.DEFINE_float("objective_threshold", 1.0,
                  "The threshold for the objective")

flags.DEFINE_boolean("adagrad", False,
                  "Use Adagrad optimizer instead")

flags.DEFINE_float("loss_epsilon", 1e-4,
                  "epsilon parameter for loss function")

flags.DEFINE_boolean("constant_lr", False,
                  "Use constant learning rate")

flags.DEFINE_boolean("wout", False,
                  "Whether we would use a separate wout")

flags.DEFINE_boolean("max_pe", False,
                  "Using maximum of partial energy instead of the sum")

flags.DEFINE_integer("max_to_keep", 5,
                     "The maximum number of checkpoint files to keep")

flags.DEFINE_boolean("normclip", False,
                    "Whether to perform norm clipping (very slow)")

flags.DEFINE_string("rep", "gm", 'The type of representation. Either gm or vec')

flags.DEFINE_integer("fixvar", 0, "whether to fix the variance or not")

FLAGS = flags.FLAGS


class Options(object):
  """Options used by our Word2MultiGauss model."""

  def __init__(self):
    # Model options.

    # Embedding dimension.
    self.emb_dim = FLAGS.embedding_size

    # Training options.
    # The training text file.
    self.train_data = FLAGS.train_data

    # The initial learning rate.
    self.learning_rate = FLAGS.learning_rate

    # Number of epochs to train. After these many epochs, the learning
    # rate decays linearly to zero and the training stops.
    self.epochs_to_train = FLAGS.epochs_to_train

    # Concurgnt training steps.
    self.concurrent_steps = FLAGS.concurrent_steps

    # Number of examples for one training step.
    self.batch_size = FLAGS.batch_size

    # The number of words to predict to the left and right of the target word.
    self.window_size = FLAGS.window_size

    # The minimum number of word occurrences for it to be included in the
    # vocabulary.
    self.min_count = FLAGS.min_count

    # Subsampling threshold for word occurrence.
    self.subsample = FLAGS.subsample

    # How often to print statistics.
    self.statistics_interval = FLAGS.statistics_interval

    # How often to write to the summary file (rounds up to the nearest
    # statistics_interval).
    self.summary_interval = FLAGS.summary_interval

    # How often to write checkpoints (rounds up to the nearest statistics
    # interval).
    self.checkpoint_interval = FLAGS.checkpoint_interval

    # Where to write out summaries.
    self.save_path = FLAGS.save_path

    #################################
    self.num_mixtures = FLAGS.num_mixtures # incorporated. needs testing

    # upper bound of norm of mu
    self.norm_cap = FLAGS.norm_cap 

    # element-wise lower bound for sigma
    self.lower_sig = FLAGS.lower_sig 

    # element-wise upper bound for sigma
    self.upper_sig = FLAGS.upper_sig 

    # whether to use spherical or diagonal covariance
    self.spherical = FLAGS.spherical   ## default to False please

    self.var_scale = FLAGS.var_scale

    self.ckpt_all = FLAGS.ckpt_all

    self.mu_scale = FLAGS.mu_scale

    self.objective_threshold = FLAGS.objective_threshold

    self.adagrad = FLAGS.adagrad

    self.loss_epsilon = FLAGS.loss_epsilon

    self.constant_lr = FLAGS.constant_lr

    self.wout = FLAGS.wout

    self.max_pe = FLAGS.max_pe

    self.max_to_keep = FLAGS.max_to_keep

    self.normclip = FLAGS.normclip

## value clipping
    self.norm_cap = FLAGS.norm_cap
    self.upper_sig = FLAGS.upper_sig
    self.lower_sig = FLAGS.lower_sig

    self.rep = FLAGS.rep

    self.fixvar = FLAGS.fixvar

class Word2GMtrainer(object):

  def __init__(self, options, session):
    self._options = options
    # Ben A: print important opts
    opts = options
    print('--------------------------------------------------------')
    print('Rep {}'.format(opts.rep))
    print('Train data {}'.format(opts.train_data))
    print('Norm cap {} lower sig {} upper sig {}'.format(opts.norm_cap,
      opts.lower_sig, opts.upper_sig))
    print('mu_scale {} var_scale {}'.format(opts.mu_scale, opts.var_scale))
    print('Num Mixtures {} Spherical Mode = {}'.format(opts.num_mixtures, opts.spherical))
    print('Emb dim {}'.format(opts.emb_dim))
    print('Epochs to train {}'.format(opts.epochs_to_train))
    print('Learning rate {} // constant {}'.format(opts.learning_rate, opts.constant_lr))
    print('Using a separate Wout = {}'.format(opts.wout))
    print('Subsampling rate = {}'.format(opts.subsample))
    print('Using Max Partial Energy Loss = {}'.format(opts.max_pe))
    print('Loss Epsilon = {}'.format(opts.loss_epsilon))
    print('Saving results to = {}'.format(options.save_path))
    print('--------------------------------------------------------')

    self._session = session
    self._word2id = {}
    self._id2word = []
    self.build_graph() # 
    self.save_vocab()

  def optimize(self, loss):
    """Build the graph to optimize the loss function."""

    # Optimizer nodes.
    # Linear learning rate decay.
    opts = self._options
    if opts.constant_lr:
      self._lr = tf.constant(opts.learning_rate)
    else:
      words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
      lr = opts.learning_rate * tf.maximum(
         0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)
      self._lr = lr
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    train = optimizer.minimize(loss,
                               global_step=self.global_step,
                               gate_gradients=optimizer.GATE_NONE)
    self._train = train

  def optimize_adam(self, loss):
    # deprecated
    opts = self._options
    # use automatic decay of learning rate in Adam
    self._lr = tf.constant(opts.learning_rate)
    self.adam_epsilon = opts.adam_epsilon
    optimizer = tf.train.AdamOptimizer(self._lr, epsilon=self.adam_epsilon)
    train = optimizer.minimize(loss, global_step=self.global_step,
                                          gate_gradients=optimizer.GATE_NONE)
    self._train = train

  def optimize_adagrad(self, loss):
    print('Using Adagrad optimizer')
    opts = self._options
    if opts.constant_lr:
      self._lr = tf.constant(opts.learning_rate)
    else:
      words_to_train = float(opts.words_per_epoch * opts.epochs_to_train)
      lr = opts.learning_rate * tf.maximum(
         0.0001, 1.0 - tf.cast(self._words, tf.float32) / words_to_train)
      self._lr = lr
    optimizer = tf.train.AdagradOptimizer(self._lr)
    train = optimizer.minimize(loss,
                               global_step=self.global_step,
                               gate_gradients=optimizer.GATE_NONE)
    self._train = train


  def calculate_loss(self, word_idxs, pos_idxs):
    # This is two methods in one (forward and nce_loss)
    self.global_step = tf.Variable(0, name="global_step")
    opts = self._options
    #####################################################
    # the model parameters
    vocabulary_size = opts.vocab_size
    embedding_size = opts.emb_dim
    batch_size = opts.batch_size

    norm_cap = opts.norm_cap
    lower_sig = opts.lower_sig
    upper_sig = opts.upper_sig

    self.norm_cap = norm_cap
    self.lower_logsig = math.log(lower_sig)
    self.upper_logsig = math.log(upper_sig)

    num_mixtures = opts.num_mixtures
    spherical = opts.spherical
    objective_threshold = opts.objective_threshold

    # the model parameters
    mu_scale = opts.mu_scale*math.sqrt(3.0/(1.0*embedding_size))
    mus = tf.Variable(tf.random_uniform([vocabulary_size, num_mixtures, embedding_size], -mu_scale, mu_scale), name='mu')
    if opts.wout:
      mus_out = tf.Variable(tf.random_uniform([vocabulary_size, num_mixtures, embedding_size], -mu_scale, mu_scale), name='mu_out')
    # This intialization makes the variance around 1
    var_scale = opts.var_scale
    logvar_scale = math.log(var_scale)
    print('mu_scale = {} var_scale = {}'.format(mu_scale, var_scale))
    var_trainable = 1-self._options.fixvar
    print('var trainable =', var_trainable)
    if spherical:
      logsigs = tf.Variable(tf.random_uniform([vocabulary_size, num_mixtures,1], 
                                              logvar_scale, logvar_scale), name='sigma', trainable=var_trainable)
      if opts.wout:
        logsigs_out = tf.Variable(tf.random_uniform([vocabulary_size, num_mixtures,1], 
                                              logvar_scale, logvar_scale), name='sigma_out', trainable=var_trainable)

    else:
      logsigs = tf.Variable(tf.random_uniform([vocabulary_size, num_mixtures, embedding_size], 
                                              logvar_scale, logvar_scale), name='sigma', trainable=var_trainable)
      if opts.wout:
        logsigs_out = tf.Variable(tf.random_uniform([vocabulary_size, num_mixtures, embedding_size], 
                                              logvar_scale, logvar_scale), name='sigma_out', trainable=var_trainable)

    mixture = tf.Variable(tf.random_uniform([vocabulary_size, num_mixtures], 0, 0), name='mixture')
    if opts.wout:
      mixture_out = tf.Variable(tf.random_uniform([vocabulary_size, num_mixtures], 0, 0), name='mixture_out')

    if not opts.wout:
      mus_out = mus
      logsigs_out = logsigs
      mixture_out = mixture

    zeros_vec = tf.zeros([batch_size], name='zeros')
    self._mus = mus
    self._logsigs = logsigs

    labels_matrix = tf.reshape(
        tf.cast(pos_idxs,
                dtype=tf.int64),
        [opts.batch_size, 1])
    # Negative sampling.
    neg_idxs, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=opts.batch_size, # Use 1 negative sample per positive sample
        unique=True,
        range_max=opts.vocab_size,
        distortion=0.75,
        unigrams=opts.vocab_counts.tolist(), name='neg_idxs'))
    self._neg_idxs = neg_idxs

    def log_energy(mu1, sig1, mix1, mu2, sig2, mix2):
      ### need to pass mix that's compatible!

      def partial_logenergy(cl1, cl2):
        m1 = mu1[:,cl1,:]
        m2 = mu2[:,cl2,:]
        s1 = sig1[:,cl1,:]
        s2 = sig2[:,cl2,:]
        with tf.name_scope('partial_logenergy') as scope:
          _a = tf.add(s1, s2) # should be do max add for stability?
          epsilon = opts.loss_epsilon

          if spherical:
            logdet = embedding_size*tf.log(epsilon + tf.squeeze(_a))
          else:
            logdet = tf.reduce_sum(tf.log(epsilon + _a), reduction_indices=1, name='logdet')
          ss_inv = 1./(epsilon + _a)
          #diff = tf.sub(m1, m2)
          diff = tf.subtract(m1, m2)
          exp_term = tf.reduce_sum(diff*ss_inv*diff, reduction_indices=1, name='expterm')
          pe = -0.5*logdet - 0.5*exp_term
          return pe

      with tf.name_scope('logenergy') as scope:
        log_e_list = []
        mix_list = []
        for cl1 in xrange(num_mixtures):
          for cl2 in xrange(num_mixtures):
            log_e_list.append(partial_logenergy(cl1, cl2))
            mix_list.append(mix1[:,cl1]*mix2[:,cl2])
        log_e_pack = tf.stack(log_e_list)
        log_e_max = tf.reduce_max(log_e_list, reduction_indices=0)

        if opts.max_pe:
          # Ben A: got this warning for max_pe
          # UserWarning:
          # Convering sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
          log_e_argmax = tf.argmax(log_e_list, dimension=0)
          log_e = log_e_max*tf.gather(mix_list, log_e_argmax)
        else:
          mix_pack = tf.stack(mix_list)
          log_e = tf.log(tf.reduce_sum(mix_pack*tf.exp(log_e_pack-log_e_max), reduction_indices=0))
          log_e += log_e_max
        return log_e
        

    def Lfunc(word_idxs, pos_idxs, neg_idxs):
      with tf.name_scope('LossCal') as scope:
        mu_embed = tf.nn.embedding_lookup(mus, word_idxs, name='MuWord')
        mu_embed_pos = tf.nn.embedding_lookup(mus_out, pos_idxs, name='MuPos')
        mu_embed_neg = tf.nn.embedding_lookup(mus_out, neg_idxs, name='MuNeg')
        sig_embed = tf.exp(tf.nn.embedding_lookup(logsigs, word_idxs), name='SigWord')
        sig_embed_pos = tf.exp(tf.nn.embedding_lookup(logsigs_out, pos_idxs), name='SigPos')
        sig_embed_neg = tf.exp(tf.nn.embedding_lookup(logsigs_out, neg_idxs), name='SigNeg')

        mix_word = tf.nn.softmax(tf.nn.embedding_lookup(mixture, word_idxs), name='MixWord')
        mix_pos = tf.nn.softmax(tf.nn.embedding_lookup(mixture_out, pos_idxs), name='MixPos')
        mix_neg = tf.nn.softmax(tf.nn.embedding_lookup(mixture_out, neg_idxs), name='MixNeg')
        
        epos = log_energy(mu_embed, sig_embed, mix_word, mu_embed_pos, sig_embed_pos, mix_pos)
        eneg = log_energy(mu_embed, sig_embed, mix_word, mu_embed_neg, sig_embed_neg, mix_neg)
        loss_indiv = tf.maximum(zeros_vec, objective_threshold - epos + eneg, name='CalculateIndividualLoss')
        loss = tf.reduce_mean(loss_indiv, name='AveLoss')

        return loss

    loss = Lfunc(word_idxs, pos_idxs, neg_idxs)
    tf.summary.scalar('loss', loss)

    return loss

  def clip_ops_graph(self, word_idxs, pos_idxs, neg_idxs):
    def clip_val_ref(embedding, idxs):
      with tf.name_scope('clip_val'):
        to_update = tf.nn.embedding_lookup(embedding, idxs)
        to_update = tf.maximum(self.lower_logsig, tf.minimum(self.upper_logsig, to_update))
        return tf.scatter_update(embedding, idxs, to_update)

    def clip_norm_ref(embedding, idxs):
      with tf.name_scope('clip_norm_ref') as scope:
        to_update = tf.nn.embedding_lookup(embedding, idxs)
        to_update = tf.clip_by_norm(to_update, self.norm_cap, axes=2)
        return tf.scatter_update(embedding, idxs, to_update)
    
    clip1 = clip_norm_ref(self._mus, word_idxs)
    clip2 = clip_norm_ref(self._mus, pos_idxs)
    clip3 = clip_norm_ref(self._mus, neg_idxs)
    clip4 = clip_val_ref(self._logsigs, word_idxs)
    clip5 = clip_val_ref(self._logsigs, pos_idxs)
    clip6 = clip_val_ref(self._logsigs, neg_idxs)

    return [clip1, clip2, clip3, clip4, clip5, clip6]

  def build_graph(self):
    """Build the graph for the full model."""
    opts = self._options
    # The training data. A text file.
    (words, counts, words_per_epoch, self._epoch, self._words, examples,
     labels) = word2vec.skipgram_word2vec(filename=opts.train_data,
                                 batch_size=opts.batch_size,
                                 window_size=opts.window_size,
                                 min_count=opts.min_count,
                                 subsample=opts.subsample)
    (opts.vocab_words, opts.vocab_counts,
     opts.words_per_epoch) = self._session.run([words, counts, words_per_epoch])
    opts.vocab_size = len(opts.vocab_words)
    print("Data file: ", opts.train_data)
    print("Vocab size: ", opts.vocab_size - 1, " + UNK")
    print("Words per epoch: ", opts.words_per_epoch)
    self._examples = examples
    self._labels = labels
    self._id2word = opts.vocab_words
    for i, w in enumerate(self._id2word):
      self._word2id[w] = i
    loss = self.calculate_loss(examples, labels)
    self._loss = loss

    if opts.normclip:
      self._clip_ops = self.clip_ops_graph(self._examples, self._labels, self._neg_idxs)

    if opts.adagrad:
      print("Using Adagrad as an optimizer!")
      self.optimize_adagrad(loss)
    else:
      # Using Standard SGD
      self.optimize(loss)
    # Properly initialize all variables.
    self.check_op = tf.add_check_numerics_ops()

    tf.initialize_all_variables().run()

    try:
      print('Try using saver version v2')
      self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep = opts.max_to_keep)
    except:
      print('Default to saver version v1')
      self.saver = tf.train.Saver(max_to_keep=opts.max_to_keep)

  def save_vocab(self):
    """Save the vocabulary to a file so the model can be reloaded."""
    opts = self._options
    with open(os.path.join(opts.save_path, "vocab.txt"), "w") as f:
      for i in xrange(opts.vocab_size):
        vocab_word = tf.compat.as_text(opts.vocab_words[i]).encode("utf-8")
        f.write("%s %d\n" % (vocab_word,
                             opts.vocab_counts[i]))

  def _train_thread_body(self):
    initial_epoch, = self._session.run([self._epoch])
    while True:
      # This is where the optimizer that minimizes loss (self._train) is run
      if not self._options.normclip:
        _, epoch = self._session.run([self._train, self._epoch])
      else:
        _, epoch, _ = self._session.run([self._train, self._epoch, self._clip_ops])
      if epoch != initial_epoch:
        break

  def train(self):
    """Train the model."""
    opts = self._options
    initial_epoch, initial_words = self._session.run([self._epoch, self._words])
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(opts.save_path, self._session.graph)
    workers = []
    for _ in xrange(opts.concurrent_steps):
      t = threading.Thread(target=self._train_thread_body)
      t.start()
      workers.append(t)
    last_words, last_time, last_summary_time = initial_words, time.time(), 0
    last_checkpoint_time = 0
    step_manual = 0

    while True:
      time.sleep(opts.statistics_interval)  # Reports our progress once a while.
      (epoch, step, loss, words, lr) = self._session.run(
          [self._epoch, self.global_step, self._loss, self._words, self._lr])
      now = time.time()
      last_words, last_time, rate = words, now, (words - last_words) / (
          now - last_time)
      print("Epoch %4d Step %8d: lr = %5.3f loss = %6.2f words/sec = %8.0f\r" %
            (epoch, step, lr, loss, rate), end="")
      sys.stdout.flush()
      if now - last_summary_time > opts.summary_interval:
        summary_str = self._session.run(summary_op)
        summary_writer.add_summary(summary_str, step)
        last_summary_time = now
      if now - last_checkpoint_time > opts.checkpoint_interval:
        self.saver.save(self._session,
                        os.path.join(opts.save_path, "model.ckpt"),
                        global_step=step.astype(int))
        last_checkpoint_time = now
      if epoch != initial_epoch:
        break
      step_manual += 1

    for t in workers:
      t.join()
    return epoch

def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)

def main(_):
  if not FLAGS.train_data or not FLAGS.save_path:
    print("--train_data and --save_path must be specified.")
    sys.exit(1)
  if not os.path.exists(FLAGS.save_path):
    print('Creating new directory', FLAGS.save_path)
    os.makedirs(FLAGS.save_path)
  else:
    print('The directory already exists', FLAGS.save_path)
  opts = Options()
  print('Saving results to {}'.format(opts.save_path))
  with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
      model = Word2GMtrainer(opts, session)
    for _ in xrange(opts.epochs_to_train):
      model.train()  
    # Perform a final save.
    model.saver.save(session,
                     os.path.join(opts.save_path, "model.ckpt"),
                     global_step=model.global_step)

if __name__ == "__main__":
  tf.app.run()
