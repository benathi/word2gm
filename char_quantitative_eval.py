import tensorflow as tf
import numpy as np
import pandas as pd
import scipy
import os
#from word2gm_loader import *
from load_model import *
from projectutil import find_list_ckpts
from ggplot import *
import re
from sklearn.metrics import f1_score

## 1. Evaluation Data Loading
def load_SimLex999(filepath='evaluation_data/SimLex-999/SimLex-999.txt'):
    _fpath = filepath if filepath is not None else os.environ['SIMLEX999_FILE']
    df = pd.read_csv(_fpath, delimiter='\t')
    word1 = df['word1'].tolist()
    word2 = df['word2'].tolist()
    score = df['SimLex999'].tolist()
    assert len(word1) == len(word2) and len(word1) == len(score)
    return word1, word2, score

def load_data_format1(filename='EN-MC-30.txt', delim='\t', verbose=False):
    if verbose: print 'Loading file', filename
    fpath = os.path.join('evaluation_data/multiple_datasets', filename)
    df = pd.read_csv(fpath, delimiter=delim, header=None)
    word1 = df[0].tolist()
    word2 = df[1].tolist()
    score = df[2].tolist()
    assert len(word1) == len(word2) and len(word1) == len(score)
    return word1, word2, score

def load_MC():
    return load_data_format1(filename='EN-MC-30.txt')

def load_MEN():
    return load_data_format1(filename='EN-MEN-TR-3k.txt', delim=' ')

def load_Mturk287():
    return load_data_format1(filename='EN-MTurk-287.txt')

def load_Mturk771():
    return load_data_format1(filename='EN-MTurk-771.txt', delim=' ')

def load_RG():
    return load_data_format1(filename='EN-RG-65.txt')

def load_RW_Stanford():
    return load_data_format1(filename='EN-RW-STANFORD.txt')

def load_WS_all():
    return load_data_format1(filename='EN-WS-353-ALL.txt')

def load_WS_rel():
    return load_data_format1(filename='EN-WS-353-REL.txt')

def load_WS_sim():
    return load_data_format1(filename='EN-WS-353-SIM.txt')

def load_YP():
    return load_data_format1(filename='EN-YP-130.txt', delim=' ')


def calculate_correlation(data_loader, metric, model, verbose=True, lower=False, option='add'):
    #### data_loader is a function that returns 2 lists of words and the scores
    #### metric is a function that takes w1, w2 and calculate the score
    print('----------------------------------------')
    word1, word2, targets = data_loader()
    distinct_words =  set(word1 + word2)
    ndistinct = len(distinct_words)
    nwords_dict = sum([w in model.word2id for w in distinct_words])
    if lower:
      nwords_dict = sum([w.lower() in model.word2id for w in distinct_words])
    #if verbose: 
    print '# of pairs {} # words total {} # words in dictionary {}({}%)'\
      .format(len(word1), ndistinct, nwords_dict, 100*nwords_dict/(1.*ndistinct))
    
    if lower:
      word1 = [word.lower() for word in word1]
      word2 = [word.lower() for word in word2]

    embs1 = model.word_to_emb(word1, option=option)
    embs2 = model.word_to_emb(word2, option=option)
    print('embs1', embs1)
    scores = np.sum(embs1*embs2, axis=1)


    spr = scipy.stats.spearmanr(scores, targets)
    if verbose: print 'Spearman correlation is {} with pvalue {}'.format(spr.correlation, spr.pvalue)
    pear = scipy.stats.pearsonr(scores, targets)
    if verbose: print 'Pearson correlation', pear
    spr_correlation = spr.correlation
    pear_correlation = pear[0]
    if np.any(np.isnan(scores)):
        spr_correlation = np.NAN
        pear_correlation = np.NAN
    return scores, spr_correlation, pear_correlation

eval_datasets = [load_SimLex999, load_WS_all, load_WS_sim, load_WS_rel, 
                load_MEN, load_MC, load_RG, load_YP,
                load_Mturk287, load_Mturk771,
                load_RW_Stanford]

eval_datasets_names_full = []
for dgen in eval_datasets:
    eval_datasets_names_full.append(dgen.__name__[5:])
eval_datasets_names = ['SL', 'WS', 'WS-S', 'WS-R', 'MEN',
                             'MC', 'RG', 'YP', 'MT-287', 'MT-771', 'RW']


# performs quantitative evaluation in a batch
def quantitative_eval(model_names, ckpt_files=None, prefix_dir='', metric_funcs = ['dot'],
  lower=False, verbose=False, char_model=True, option='add'):
  # model_names is a list of pairs (model_abbreviation, save_path)
  # ckpt_file is a list of the same length as model_names, if not None
  assert ckpt_files is None or len(ckpt_files) == len(model_names)
  spearman_corrs = pd.DataFrame()
  spearman_corrs['Dataset'] = eval_datasets_names
  # folder path of this code
  # allow it to be called from other directory
  dir_path = os.path.dirname(os.path.realpath(__file__))
  for i, (model_abbrev, save_path) in enumerate(model_names):
      if verbose: print 'Processing', save_path
      if True:
          if verbose: print 'dir path =', dir_path
          save_path_full = os.path.join(dir_path, prefix_dir, save_path)
          #ckpt_file = None if ckpt_files is None else ckpt_files[i]
          #w2mg = Word2GM(save_path_full, ckpt_file=ckpt_file, verbose=verbose)
          model_emb = WordEmb(save_path_full, char_model=char_model)
          for metric_name in metric_funcs:
              results = []
              if verbose: print 'metric', metric_name
              for dgen in eval_datasets:
                  if verbose: print 'data', dgen.__name__
                  _, sp, pe = calculate_correlation(dgen, metric_name, model_emb, lower=lower, verbose=verbose, option=option)
                  #print scores
                  results.append(sp*100)
              colname = '{}/{}'.format(model_abbrev, metric_name)
              spearman_corrs[colname] = results
  return spearman_corrs


if __name__ == '__main__':
  #model_names = [('1', 'modelfiles/model_word_char_v2-no_gs')]
  model_names = [('2', 'modelfiles/model_word_only')]
  print(quantitative_eval(model_names, char_model=False, option='dict'))