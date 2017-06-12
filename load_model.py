from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pickle, os, re
from word2gm_trainer_repvec import Options # Isn't this just a namespace?
from ops import batch_norm, highway
from TDNN import TDNN
import numpy as np
# We need to load (1) The vector rep
# (2) The character model

# Some library has quite a seemless model loading in Tensorflow
# 

# TODO - might need to import some methods from word2gm_trainer_repvec 
# TODO - need to load opts too. This is crucial for model construction.
# 

class WordEmb(object):
  def __init__(self, save_path, session=None, load=True):
    # load the models
    self.save_path = save_path

    # loading options
    options_path = os.path.join(self.save_path, 'options.p')
    self.options = pickle.load(open(options_path, 'r'))
    # loading vocab
    self.load_vocab()
    # determine the latest ckpt file
    latest_ckpt_file = tf.train.latest_checkpoint(self.save_path)
    print('The latest ckpt file ', latest_ckpt_file)

    # load the model
    if session is None:
      self.session = tf.Session()
    else:
      self.session = session
    self.load_model()
    saver = tf.train.Saver(tf.global_variables())
    if load:
      saver.restore(self.session, latest_ckpt_file)
    else:
      tf.global_variables_initializer().run(session=self.session)

  def load_vocab(self):
    id2word = [''.join([i if ord(i) < 128 
            else '' for i in 
            re.match(r'(.+)\s([\d]+)\s', line).group(1)])
            for line in open(os.path.join(self.save_path, 'vocab.txt'), 'r')
            ]
    self.vocab_size = len(id2word)
    #assert len(id2word) == self.vocab_size, \
    #        'Expecting vocab size to match ckpt:{} vocab.txt{}'.format(self.vocab_size, len(id2word))
    self.id2word = id2word
    word2id = {}
    for _i in xrange(self.vocab_size):
        word2id[id2word[_i]] = _i
    self.word2id = word2id


  def load_model(self):
    opts = self.options
    
    # (1) subword embeddings
    with tf.variable_scope("char_emb_all") as scope1:
      with tf.variable_scope("CHAR_MODEL") as scope:
        char_inputs = tf.placeholder(tf.int32, [None, opts.max_word_len], name='char_inputs')
        char_W = tf.get_variable("char_embed",
                [opts.char_vocab_size, opts.char_embed_dim])
        
        char_embed = tf.nn.embedding_lookup(char_W, char_inputs)

        if opts.char_emb_type == 'bilstm':
          assert False, 'Bi LSTM model not implemented'
          pass
        elif opts.char_emb_type == 'cnn':
          print('Using the CNN Model for Character Embedding')
          # The feature maps is configurable but fixed for now
          #feature_maps = [50, 100, 150, 200, 200, 200, 200] # This adds up to 1100
          feature_maps = [50,50,50,50,50,50,50]
          dim = sum(feature_maps)
          kernels = [1,2,3,4,5,6,7]
          char_cnn = TDNN(char_embed, embed_dim=opts.char_embed_dim, feature_maps=feature_maps, kernels=kernels,
            max_seq_len=opts.max_word_len)
          cnn_output = char_cnn.output
          cnn_output = tf.reshape(cnn_output, [-1, dim])
          if opts.use_batch_norm:
            bn = batch_norm(dim=dim)
            cnn_output = bn(cnn_output)
          if opts.use_highway:
            cnn_output = highway(cnn_output, size=dim, layer_size=1, bias=0)
          output = cnn_output
    # now we have output for the char embedding model
    # (2) word embeddings
    init_width = 0.5 / opts.emb_dim
    emb = tf.Variable(
        tf.random_uniform(
            [opts.vocab_size, opts.emb_dim], -init_width, init_width),
        name="emb")

    # This is wout?
    sm_w_t = tf.Variable(
        tf.zeros([opts.vocab_size, opts.emb_dim]),
        name="sm_w_t")

    self.emb = emb
    self.char_inputs = char_inputs
    self.char_output = output

  # BenA: make sure this is consistent with the model's
  # maybe add this to options
  def char_to_idx(self, char):
    idx_raw = ord(char)
    if idx_raw >= 33 and idx_raw <= 126:
      return ord(char) - 33 + 1
    else:
      return 0

  def dictionary_embedding(self, word_list):
    indicator = [word in self.word2id for word in word_list]
    word_sublist = [word for word in word_list if word in self.word2id]
    dict_emb_sublist = self.dictionary_embedding_indict(word_sublist)
    dict_emb = np.zeros((len(word_list), dict_emb_sublist.shape[1]))
    counter = 0
    for i, val in enumerate(indicator):
      if val:
        dict_emb[i] = dict_emb_sublist[counter]
        counter += 1
    print('Counter ending at ', counter)
    assert counter == dict_emb_sublist.shape[0]
    return dict_emb

  def dictionary_embedding_indict(self, word_list):
    # if not in there, maybe return 0 vector?
    for word in word_list:
      assert word in self.word2id, 'Word {} is not in the dictionary'
    word_idxs = np.array([self.word2id[word] for word in word_list], dtype=np.int32)
    word_idxs_tf = tf.constant(word_idxs, dtype=tf.int32)
    dict_embs_tf = tf.nn.embedding_lookup(self.emb, word_idxs_tf)
    #with tf.Session() as sess:
    #dict_embs = dict_embs_tf.eval()
    dict_embs = self.session.run(dict_embs_tf)
    return dict_embs

  def get_char_input(self, word_list):
    char_input = np.zeros((len(word_list), self.options.max_word_len), dtype=np.int32)
    for i, word in enumerate(word_list):
      word_seq = np.array([self.char_to_idx(c) for c in word])
      char_input[i, :min(len(word_seq), self.options.max_word_len)] = word_seq[:self.options.max_word_len]
    return char_input

  def char_embedding(self, word_list):
    # assume that we take the words directly
    char_input = self.get_char_input(word_list)
    print('char input=', char_input)
    char_embs = self.session.run(self.char_output, feed_dict={self.char_inputs:char_input})
    return char_embs

  def word_to_emb(self, word_list, option='add'):
    assert option in ['add', 'char', 'dict']
    # Take a list of strings in a minibatch and obtain the embeddings
    # (1) check if it's in the dictionary
    # (2) obtain character sequence and get the char embedding sequence
    # (3) feed char embedding sequence to obtain the word embedding

    # now only works if all words in the dictionary
    if option == 'add':
      return self.char_embedding(word_list) + self.dictionary_embedding(word_list)
    elif option == 'char':
      return self.char_embedding(word_list)
    elif option == 'dict':
      return self.dictionary_embedding(word_list)
    elif option == 'both':
      return [self.char_embedding(word_list), self.dictionary_embedding(word_list)]

if __name__ == '__main__':
  # BenA: loading seems to do something for sure.
  # Why does the char emb turn out to be all 1 and -1?
  m = WordEmb(save_path='modelfiles/model_word_char_v2-no_gs/')
  print('Dictionary Embedding')
  print(m.dictionary_embedding(['dog', 'cat']).shape)
  print('Char Embedding')
  print(m.char_embedding(['dog', 'cat']))
  print(m.char_embedding(['dog', 'cat']).shape)





















