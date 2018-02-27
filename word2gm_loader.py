import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import os
word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))
import os, re
import operator
import sys
import pandas as pd
from ggplot import * # TODO - make this compatible

# Retrict to CPU only
os.environ["CUDA_VISIBLE_DEVICES"]=""

class Word2GM(object):
    def __init__(self, save_path, ckpt_file=None, verbose=True):
        # create a new session and a new graph every time this object is constructed
        # if a ckpt file is not provided, use the latest ckpt file.
        self.ckpt_file = ckpt_file 
        self.logdir = save_path
        with tf.Graph().as_default() as g:
            with tf.Session(graph=g) as session:
                self.save_path = save_path
                self.session = session
                self.load_model(verbose)
                self.load_vocab()

    def load_vocab(self):
        id2word = [''.join([i if ord(i) < 128 
            else '' for i in 
            re.match(r'(.+)\s([\d]+)\s', line).group(1)])
            for line in open(os.path.join(self.save_path, 'vocab.txt'), 'r')
            ]
        assert len(id2word) == self.vocab_size, \
                'Expecting vocab size to match ckpt:{} vocab.txt{}'.format(self.vocab_size, len(id2word))
        self.id2word = id2word
        word2id = {}
        for _i in xrange(self.vocab_size):
            word2id[id2word[_i]] = _i
        self.word2id = word2id

    def load_model(self, verbose=True):
        latest_ckpt_file = tf.train.latest_checkpoint(self.save_path) if self.ckpt_file is None else self.ckpt_file
        if verbose and self.ckpt_file is None:
            print('Using the latest checkpoint file', latest_ckpt_file)
        elif verbose:
            print('Using the provided checkpoint file: ', self.ckpt_file)

        meta_graph_path = latest_ckpt_file + '.meta'
        new_saver = tf.train.import_meta_graph(meta_graph_path)
        new_saver.restore(self.session, latest_ckpt_file)

        [mus, logsigs] = self.session.run(['mu:0', 'sigma:0'])
        self.num_mixtures = 1 if len(mus.shape) == 2 else mus.shape[1]
        self.vocab_size = mus.shape[0]
        if verbose: print('Number of mixtures = ', self.num_mixtures)

        # handles support for > 2 (softmax case) later!
        if self.num_mixtures >= 2:
            #: if num_mixtures = 1 but mus.shape is 3 dim, then it's a new code
            # this is handled by the softmax case (even though it's 1 dimensional)
            [mixture_score] = self.session.run(['mixture:0'])
            self.word_dim = mus.shape[2]
            ## store vars
            self.mus = np.copy(mus)
            self.logsigs = np.copy(logsigs)
            if len(mixture_score.shape) == 1:
                # word2mixgauss code
                assert self.num_mixtures == 2
                # This is for word2mixgauss code: do sigmoid and expand to 2 dim
                self.mixture = np.ones((self.vocab_size, self.num_mixtures))
                self.mixture[:,0] = 1.0/(1.0 + np.exp(-mixture_score))
                self.mixture[:,1] = 1.0 - self.mixture[:,0]
            else:
                # This is for word2multigauss code: do a softmax
                assert len(mixture_score.shape) == 2 and mixture_score.shape[1] == self.num_mixtures
                # calculate softmax
                diff_exp = np.exp(mixture_score - np.max(mixture_score, axis=1, keepdims=True))
                self.mixture = diff_exp/np.sum(diff_exp, axis=1, keepdims=True)
        else:
            # In this case, num_mixures = 1: it can be either the old model and the new model
            assert self.num_mixtures == 1, 'Expecting 1 mixture'
            #assert len(mus.shape) == 2, 'Expecting mus to be a 2-d array'
            #assert len(logsigs.shape) == 2, 'Expectging logsigs to be a 2-d array'
            if len(mus.shape) == 2 and len(logsigs.shape) == 2:
                # for word2gauss code
                #print('Here!')
                self.word_dim = mus.shape[1]
                self.mus = np.copy(np.expand_dims(mus, axis=1))
                self.logsigs = np.copy(np.expand_dims(logsigs, axis=1))
            elif len(mus.shape) == 3 and len(logsigs.shape) == 3:
                self.word_dim = mus.shape[2]
                self.mus = np.copy(mus)
                self.logsigs = np.copy(logsigs)
            else:
                assert False, 'Unexpected error'
            self.mixture = np.ones((self.vocab_size, self.num_mixtures))


        # normalized mus
        norm_mu = np.linalg.norm(self.mus, axis=2, keepdims = True)
        self.mus_n_multi = self.mus/norm_mu
        self.mus_n = np.reshape(self.mus_n_multi, 
                (self.vocab_size*self.num_mixtures, self.word_dim),
                order='C')
        # This might be incorrect for spherical case
        # need to be logsig *
        self.detA = np.sum(self.logsigs, axis=2)
        self.detA = np.reshape(self.detA, (self.vocab_size*self.num_mixtures,), order = 'C')
        ## end of load_model

    #####
    def find_nearest_neighbors(self, idx, cl):
        # idx is the word id
        # cl is the cluster
        dist = np.dot(self.mus_n, self.mus_n[idx*self.num_mixtures + cl])
        sorted_idxs = dist.argsort()[::-1]
        return sorted_idxs

    def idxs2words(self, idxs):
        # convert a list of strings to a list of words
        words = ["{}:{}".format(self.id2word[idx/self.num_mixtures], idx%self.num_mixtures) for idx in idxs]
        return words

    def sort_low_var(self, idxs):
        # given a list of indices (linear), sort elements with lowest variance first
        list_pair = [(idx, self.detA[idx]) for idx in idxs]
        list_pair.sort(key=operator.itemgetter(1))
        # return simply the indices as well as the list of idx-variance pairs
        return [p[0] for p in list_pair], list_pair

    def show_nearest_neighbors(self, idx_or_word, cl=0, num_nns=20, plot=True, verbose=False):
        assert isinstance(idx_or_word, int) or idx_or_word in self.word2id, 'Provide index or word in vocabulary'
        idx = idx_or_word
        if idx_or_word in self.word2id:
            idx = self.word2id[idx_or_word]
        dist = np.dot(self.mus_n, self.mus_n[idx*self.num_mixtures + cl])
        highsim_idxs = dist.argsort()[::-1]
        # select top num_nns (linear) indices with the highest cosine similarity
        highsim_idxs = highsim_idxs[:num_nns]
        dist_val = dist[highsim_idxs]
        words = self.idxs2words(highsim_idxs)
        var_val = np.array([self.detA[_idx] for _idx in highsim_idxs])
        # plot all the words
        if plot:
            df = pd.DataFrame()
            df['text'] = words
            df['sim'] = dist_val
            df['logvar'] = var_val
            mix = self.mixture[idx, cl]
            plot = (ggplot(aes(x='sim', y='logvar', label='text'), data=df)
                    + geom_point(size=5)
                    + geom_text(size=10)
                    + ggtitle("Neighbors of [{}:{}] with mixture probability {:.4g}".format(self.id2word[idx], cl, mix))
                    )
            print plot
        print 'Top 10 highest similarity'
        print words[:10]
        if verbose: print dist_val[:10]
        print 'Top 10 lowest variance of top {} highest similarity'.format(num_nns)
        low_var_idxs, var_val = self.sort_low_var(highsim_idxs)
        print self.idxs2words(low_var_idxs)
        if verbose: print var_val

    def words_to_idxs(self, word_list, discard_unk=False, verbose=False):
        assert isinstance(word_list, list), 'Expected a list'
        if discard_unk:
            return self.words_to_idxs_discard_unk(word_list)
        else:
            return [self.get_idx(_w, verbose) for _w in word_list]

    def words_to_idxs_discard_unk(self, word_list):
        idxs =  [self.word2id[word] for word in word_list if word in self.word2id]
        if len(idxs) == 0:
            return [0] # return the index of unknown
        return idxs

    def get_idx(self, word, verbose=False):
        if word in self.word2id:
            return self.word2id[word]
        else:
            if verbose: print 'Unknown word [{}]'.format(word)
            return 0
    #### 
    def dot(self, idx1, cl1, idx2, cl2):
        _res = np.dot(self.mus_n_multi[idx1, cl1], self.mus_n_multi[idx2, cl2])
        return _res

    def maxdot(self, idx1, idx2, verbose=False):
        metric_grid = np.zeros((self.num_mixtures, self.num_mixtures))
        for cl1 in range(self.num_mixtures):
            for cl2 in range(self.num_mixtures):
                metric_grid[cl1, cl2] = self.dot(idx1, cl1, idx2, cl2)
                if verbose: print metric_grid
        return np.max(metric_grid)

    def avedot(self, idx1, idx2, verbose=False):
        metric_grid = np.zeros((self.num_mixtures, self.num_mixtures))
        for cl1 in range(self.num_mixtures):
            for cl2 in range(self.num_mixtures):
                metric_grid[cl1, cl2] = self.dot(idx1, cl1, idx2, cl2)
                if verbose: print metric_grid
        return np.mean(metric_grid)

    def negkl(self, w1, cl1, w2, cl2):
        ## This is for KL and min KL
        # This is -2*KL(w1 || w2)
        D = len(self.mus_n_multi[0,0])
        # note: ignore -D because it's a constant
        m1 = self.mus[w1, cl1]
        m2 = self.mus[w2, cl2]
        epsilon = 1e-4
        logsig1 = self.logsigs[w1, cl1]
        logsig2 = self.logsigs[w2, cl2]
        sig1 = np.exp(logsig1)
        sig2 = np.exp(logsig2)
        s2_inv = 1./(epsilon + sig2)

        sph = (len(logsig1) == 1)

        #print 'D = {} Spherical = {}'.format(D, sph)

        diff = m1 - m2
        exp_term = np.sum(diff*s2_inv*diff)

        if sph:
            tr_term = D*sig1*s2_inv
        else:
            tr_term = np.sum(sig1*s2_inv)
        

        if sph:
            log_rel_det = D*logsig1 - D*logsig2
        else:
            log_rel_det = np.sum(logsig1 - logsig2)

        res =  tr_term + exp_term - log_rel_det
        return -res

    def max_negkl(self, idx1, idx2, verbose = False):
        metric_grid = np.zeros((self.num_mixtures, self.num_mixtures))
        for cl1 in range(self.num_mixtures):
            for cl2 in range(self.num_mixtures):
                metric_grid[cl1, cl2] = self.negkl(idx1, cl1, idx2, cl2)
                if verbose: print metric_grid
        return np.max(metric_grid)


    #### compute the norm of the difference
    def norm(self, idx1, cl1, idx2, cl2):
        _res = np.linalg.norm(self.mus[idx1, cl1] - self.mus[idx2, cl2])
        return _res

    # it actually should be the negative of minimum norm
    def maxnorm(self, idx1, idx2, verbose=False):
        # returns the negative max norm
        metric_grid = np.zeros((self.num_mixtures, self.num_mixtures))
        for cl1 in range(self.num_mixtures):
            for cl2 in range(self.num_mixtures):
                metric_grid[cl1, cl2] = self.norm(idx1, cl1, idx2, cl2)
                if verbose: print metric_grid
        return -np.min(metric_grid)

    def disdot(self, w1, w2):
    	num_mix = self.num_mixtures
    	mu1 = self.mus[w1]
    	mu2 = self.mus[w2]
    	sigma1 = np.exp(self.logsigs[w1])
        sigma2 = np.exp(self.logsigs[w2])
        mix1 = self.mixture[w1]
        mix2 = self.mixture[w2]
        def partial_energy(cl1, cl2):
            # cl1, cl2 are 'cluster' indices
            _a = sigma1[cl1] + sigma2[cl2]
            _res = -0.5*np.sum(np.log(_a))
            ss_inv = 1./_a
            diff = mu1[cl1] - mu2[cl2]
            _res += -0.5*np.sum(
                diff*ss_inv*diff
            )
            return _res
    
        partial_energies = np.zeros((num_mix, num_mix))
        for _i in range(num_mix):
            for _j in range(num_mix):
                partial_energies[_i,_j] = partial_energy(_i, _j)
    
        # for numerical stability
        max_partial_energy = np.max(partial_energies)
        #print 'max partial (log) energy', max_partial_energy
        energy = 0
        for _i in range(num_mix):
            for _j in range(num_mix):
                energy += \
                mix1[_i]*mix2[_j]*np.exp(partial_energies[_i,_j] - max_partial_energy)
        log_energy = max_partial_energy + np.log(energy)
        return log_energy

    # this is to determine the best cluster based on context
    def find_best_cluster(self, w, context, verbose=False, criterion='max'):
        assert criterion in ['max', 'mean', 'mean_of_max']
        scores = np.zeros((self.num_mixtures))
        for i in range(self.num_mixtures):
            all_scores = np.zeros((len(context), self.num_mixtures))
            for j, context_word in enumerate(context):
                for context_cl in range(self.num_mixtures):
                    all_scores[j, context_cl] = self.dot(w, i, context_word, context_cl)
            if criterion == 'max':
                scores[i] = np.max(all_scores)
            elif criterion == 'mean':
                scores[i] = np.mean(all_scores)
            elif criterion == 'mean_of_max':
                max_scores = np.max(all_scores, axis=1)
                if verbose:
                    print 'max scores', max_scores
                assert len(max_scores) == len(context)
                scores[i] = np.mean(max_scores)

            if verbose:
                print 'Mixture ', i 
                print 'all scores = {} with aggregate score = {}'.format(all_scores, scores[i])
        cl_max = np.argmax(scores)
        return cl_max

    def wordsim_context(self, w1, c1, w2, c2, metric='dot_context', criterion='max', verbose=False):
        assert metric in ['dot_context', 'maxdot', 'avedot']
        # w1 is a word
        # c1 is a list of words

        w1 = self.get_idx(w1)
        w2 = self.get_idx(w2)

        if w1 == w2:
            return 1.0

        if metric == 'dot_context':
            if verbose: print 'Using dot context'
            c1 = self.words_to_idxs(c1, discard_unk=True)
            c2 = self.words_to_idxs(c2, discard_unk=True)
            cl1 = self.find_best_cluster(w1, c1, criterion=criterion, verbose=verbose)
            cl2 = self.find_best_cluster(w2, c2, criterion=criterion, verbose=verbose)
            score = self.dot(w1, cl1, w2, cl2)
            return score
        elif metric == 'maxdot':
            if verbose: print 'Using maxdot'
            score = self.maxdot(w1, w2, verbose=verbose)
            return score
        elif metric == 'avedot':
            if verbose: print 'Using avedot'
            score = self.avedot(w1, w2, verbose=verbose)

    def visualize_embeddings(self, port=6006, call_tensorboard=False):
        from tensorflow.contrib.tensorboard.plugins import projector
        from subprocess import call
        mus = self.mus
        vocabs = self.id2word
        mus = np.resize(mus, (mus.shape[0]*mus.shape[1], mus.shape[2]))
        labels = []
        for word in vocabs:
            for i in range(self.num_mixtures):
                labels.append(word+":{}".format(i))
        emb_logdir = self.logdir + '_emb'

        if not os.path.exists(emb_logdir):
            os.makedirs(emb_logdir)
        else:
            print 'The directory already exists!'
        thefile = open(emb_logdir + '/labels.csv', 'w')
        for item in labels:
            thefile.write("%s\n" % item)
        with tf.Graph().as_default() as g:
            with tf.Session(graph=g) as session:
                embedding_var = tf.Variable(mus, name='mus')
                init = tf.initialize_all_variables()
                init.run()
                saver = tf.train.Saver()
                saver.save(session, os.path.join(emb_logdir, "model.ckpt"), 0)
                summary_writer = tf.train.SummaryWriter(emb_logdir)
                config = projector.ProjectorConfig()
                embedding = config.embeddings.add()
                embedding.tensor_name = embedding_var.name
                embedding.metadata_path = os.path.join(emb_logdir, 'labels.csv')
                projector.visualize_embeddings(summary_writer, config)
        if call_tensorboard:
            call(["tensorboard", "--logdir={}".format(emb_logdir)])


if __name__=='__main__':
    sess = tf.Session()
    word2mixgauss = Word2GM(save_path='modelfiles/t8-2s-e10-v05-lr05d-mc100-ss5-nwout-adg-win10', session=sess)
    word2mixgauss.show_nearest_neighbors('the', 0, 20)
