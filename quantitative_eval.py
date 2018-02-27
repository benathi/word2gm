import tensorflow as tf
import numpy as np
import pandas as pd
import scipy
import os
from word2gm_loader import *
from projectutil import find_list_ckpts
from ggplot import *
import re
from sklearn.metrics import f1_score

## 1. Evaluation Data Loading
def load_SimLex999(filepath='evaluation_data/SimLex-999/SimLex-999.txt'):
    #_fpath = filepath if filepath is not None else os.environ['SIMLEX999_FILE']
    _fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
      'evaluation_data/SimLex-999/SimLex-999.txt')
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

def calculate_correlation(data_loader, metric, w2g, verbose=True, lower=False):
    #### data_loader is a function that returns 2 lists of words and the scores
    #### metric is a function that takes w1, w2 and calculate the score
    list_metrics = ['ave', 'max', 'dis', 'maxnorm']
    assert metric in list_metrics, \
            'Please choose a valid metric option in {}'.format(list_metrics)
    if metric == 'ave':
        metric = w2g.avedot
    elif metric == 'max':
        metric = w2g.maxdot
    elif metric == 'dis':
        metric = w2g.disdot
    elif metric == 'maxnorm':
        metric = w2g.maxnorm
        
    word1, word2, targets = data_loader()
    distinct_words =  set(word1 + word2)
    ndistinct = len(distinct_words)
    nwords_dict = sum([w in w2g.word2id for w in distinct_words])
    if lower:
      nwords_dict = sum([w.lower() in w2g.word2id for w in distinct_words])
    if verbose: print '# of pairs {} # words total {} # words in dictionary {}({}%)'\
    .format(len(word1), ndistinct, nwords_dict, 100*nwords_dict/(1.*ndistinct))
    
    if lower:
      word1 = [word.lower() for word in word1]
      word2 = [word.lower() for word in word2]

    word1_idxs = w2g.words_to_idxs(word1)
    word2_idxs = w2g.words_to_idxs(word2)
    scores = np.zeros((len(word1_idxs)))
    for _i, [w1, w2] in enumerate(zip(word1_idxs, word2_idxs)):
        scores[_i] = metric(w1, w2)
    #scores = np.zeros((len(targets)))
    #print 'scores', scores
    #print 'targets', targets        
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
def quantitative_eval(model_names, ckpt_files=None, prefix_dir='', metric_funcs = ['max', 'dis'],
  lower=False, verbose=False):
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
          ckpt_file = None if ckpt_files is None else ckpt_files[i]
          w2mg = Word2GM(save_path_full, ckpt_file=ckpt_file, verbose=verbose)
          for metric_name in metric_funcs:
              results = []
              if verbose: print 'metric', metric_name
              for dgen in eval_datasets:
                  if verbose: print 'data', dgen.__name__
                  _, sp, pe = calculate_correlation(dgen, metric_name, w2mg, lower=lower, verbose=verbose)
                  #print scores
                  results.append(sp*100)
              colname = '{}/{}'.format(model_abbrev, metric_name)
              spearman_corrs[colname] = results
  return spearman_corrs

def to_tex(spearman_corrs, list_columns=None, new_column_names=None):
  reporting_results = spearman_corrs
  if list_columns is not None:
    reporting_results = reporting_results[ ['Dataset'] + list_columns]
  if new_column_names is not None:
    reporting_results.columns = ['Dataset'] + new_column_names
  latex_version = reporting_results.to_latex(index=False, float_format=lambda _f: '{0:.4g}'.format(_f))
  print latex_version

def quanteval_plot_ind(model_folder, prefix_dir='', lower=False, verbose=False,
  debug=False):
  # plotting many scores over time for word similarity

  ckpt_nums, ckpt_names = find_list_ckpts(model_folder, prefix_dir=prefix_dir)
  if debug:
    ckpt_nums = ckpt_nums[:3]
    ckpt_names = ckpt_names[:3]
  scores_long = []
  ckpt_nums_long = []
  labels_long = []
  for ckpt_num, ckpt_name in zip(ckpt_nums, ckpt_names):
      #print ckpt_name
      sp_corrs = quantitative_eval([(model_folder, model_folder)], 
        ckpt_files=[ckpt_name], prefix_dir=prefix_dir,
        metric_funcs = ['max'], # we should allow the ability to change this
        lower=lower,
        verbose=verbose)
      
      scores = sp_corrs[model_folder + '/max'].tolist()
      scores_long = scores_long + scores
      ckpt_nums_long = ckpt_nums_long + len(eval_datasets_names)*[ckpt_num]
      labels_long = labels_long + eval_datasets_names

      # Next, add evaluation for SCWS
      df_sp = quantitative_scws_df(model_folder, prefix_dir, ckpt_file=ckpt_name, verbose=verbose)
      labels_long += df_sp['method'].tolist()
      ckpt_nums_long += len(df_sp)*[ckpt_num]
      scores_long += df_sp['spearman'].tolist()

      # Next, add the average of all scores
      labels_long += ['AVERAGE']
      ckpt_nums_long += [ckpt_num]
      scores_long += [np.mean(np.array(scores))] # add more scores

  df = pd.DataFrame()
  df['x'] = ckpt_nums_long
  df['scores'] = scores_long
  df['dataset'] = labels_long
  plot = (ggplot(aes(x='x', y='scores', color='dataset'), data=df)
                    + geom_point(size=5)
                    + geom_line()
                    + ggtitle("Scores as time progress")
                    )
  return plot, df

def quantitative_eval_over_time(model_folder, prefix_dir='', lower=False):
  # This is using max cosine similarity
  ckpt_nums, ckpt_names = find_list_ckpts(model_folder, prefix_dir=prefix_dir)
  scores = []
  for ckpt_num, ckpt_name in zip(ckpt_nums, ckpt_names):
      print ckpt_name
      sp_corrs = quantitative_eval([(model_folder, model_folder)], [ckpt_name], prefix_dir=prefix_dir,
        lower=lower)
      sum_score = sum(sp_corrs[model_folder + '/max'])
      scores.append(sum_score)
  df = pd.DataFrame()
  df['x'] = ckpt_nums
  df['scores'] = scores
  plot = (ggplot(aes(x='x', y='scores'), data=df)
                    + geom_point(size=5)
                    + geom_line()
                    + ggtitle("Scores as time progress")
                    )
  return plot, df

def process_huang(filename='ehuang_sim_wcontext/SCWS/ratings.txt',
                context_window=5,
                verbose=False):
  dirname = 'evaluation_data'
  filepath = os.path.join(dirname, filename)
  f = open(filepath, 'r')
  result_list = []
  for line_num, line in enumerate(f):
    ob = re.search(r'(.*)<b>(.*)</b>(.*)<b>(.*)</b>(.*?)\t(.+)', line)
    pre1 = ob.group(1).split()
    word1 = ob.group(2).strip()
    middle = ob.group(3).split()
    word2 = ob.group(4).strip()
    post2 = ob.group(5).split()
    scores = ob.group(6).split()
        
    pre1 = pre1[-context_window:]
    post1 = middle[:context_window]
    pre2 = middle[-context_window:]
    post2 = post2[:context_window]
        
    scores = [float(score) for score in scores]
    ave_score = np.mean(np.array(scores))
        
    if verbose:
      print line
      print '---------'
      print 'word {} has context'.format(word1)
      print pre1
      print post1
      print '.........'
      print 'word {} has context'.format(word2)
      print pre2
      print post2
      print 'scores = ', scores
      print 'average score = ', ave_score
    result = (word1, pre1+post1, word2, pre2+post2, ave_score)
    result_list.append(result)
  return result_list

# returns a dataframe of results
def quantitative_scws_df(save_path, prefix_dir='', ckpt_file=None, verbose=False):
  # run all metrics and criteria
  dir_path = os.path.dirname(os.path.realpath(__file__))
  save_path_full = os.path.join(dir_path, prefix_dir, save_path)
  w2mg = Word2GM(save_path_full, ckpt_file=ckpt_file, verbose=verbose)

  sp1, _ = quantitative_scws(w2mg, prefix_dir, metric='maxdot', criterion='',
    verbose=verbose, lower=False)

  # return in dataframe format
  df = pd.DataFrame()
  df['method'] = ['SCWS_maxdot']
  df['spearman'] = [sp1.correlation*100]
  return df

# helper function
def quantitative_scws(model, prefix_dir='model_files',
    metric='dot_context', criterion='max', verbose=False, lower=False):
  # quantitative evaluation using word similarity with sentential context
  # model can be either w2mg or save path
  if type(model) is str:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_path_full = os.path.join(dir_path, '' + prefix_dir, model)
    w2mg = Word2GM(save_path_full)
  else:
    w2mg = model
  data_huang = process_huang()
  #ws1, cs1, ws2, cs2, scores = zip(*data_huang)
  model_scores = []
  human_scores = []
  df = pd.DataFrame()

  for i, (w1, c1, w2, c2, human_score) in enumerate(data_huang):
    human_scores.append(human_score)
    if lower:
      w1, w2 = [w1.lower(), w2.lower()]
    model_score = w2mg.wordsim_context(w1, c1, w2, c2, 
      metric=metric, criterion=criterion, verbose=verbose)
    model_scores.append(model_score)

  df['word1'], _, df['word2'], _, df['human scores'] = zip(*data_huang)
  df['model scores'] = model_scores
  if verbose:
    print df

  # compute spearman correlation
  spr = scipy.stats.spearmanr(model_scores, human_scores)
  return spr, df

# data format:
def load_entailment_baroni12():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    entailment_prefix = 'evaluation_data/entailment_baroni2012'
    pos_examples_filename = 'positive-examples.txtinput'
    neg_examples_filename = 'negative-examples.txtinput'
    pos_path = os.path.join(dir_path, entailment_prefix, pos_examples_filename)
    neg_path = os.path.join(dir_path, entailment_prefix, neg_examples_filename)
    #print pos_path
    def load_entailment(fname):
      wlist1 = []
      wlist2 = []
      for line in open(fname, 'r'):
        w1, w2 = line.split()
        # remove '-n' at the end
        w1 = w1[:-2]
        w2 = w2[:-2]
        wlist1.append(w1)
        wlist2.append(w2)
      return wlist1, wlist2

    wlist1, wlist2 = load_entailment(pos_path)
    _wlist1, _wlist2 = load_entailment(neg_path)
    wlist1 += _wlist1
    wlist2 += _wlist2
    assert len(wlist1) == len(wlist2)
    return wlist1, wlist2

def calculate_entailment(model, prefix_dir='', metric='maxdot', verbose=False, reverse=False):
  # do distance = maxdot, KL
  # use both F1 and AP
    if type(model) is str:
      dir_path = os.path.dirname(os.path.realpath(__file__))
      save_path_full = os.path.join(dir_path, '' + prefix_dir, model)
      w2mg = Word2GM(save_path_full, verbose=verbose)
    else:
      w2mg = model
    wlist1, wlist2 = load_entailment_baroni12()
    labels = np.array(len(wlist1)/2*[1] + len(wlist1)/2*[0])
    scores = np.array(calculate_scores_entailment(w2mg, wlist1, wlist2, metric))
    if reverse:
      scores = np.array(calculate_scores_entailment(w2mg, wlist2, wlist1, metric))
    # Find the best precision that maximizes
    precs = []
    f1s = []

    if verbose: print scores

    search_space = None
    if metric == 'maxdot':
      search_space = np.linspace(0,1,200)
    elif metric == 'kl':
      search_space = np.linspace(-100,0,2000)

    for thres in search_space:
      if verbose: print 'Threshold = ', thres
      thres_array = np.array(len(labels)*[thres])
      num_agree = np.sum(labels == (np.array(scores) > thres_array))
      num_above = np.sum((np.array(scores) > thres_array))
      precs.append(num_agree/(1.*len(labels)))
      f1 = f1_score(labels, np.array(scores) > thres_array)
      f1s.append(f1)
      if verbose: print 'num above = {} num agree = {} f1 = {}'.format(num_above, num_agree, f1)
    if verbose: print precs
    best_prec = np.max(np.array(precs))
    best_f1 = np.max(np.array(f1s))
    if verbose: print 'Best precision {} Best F1 {}'.format(best_prec, best_f1)
    return best_prec, best_f1

def calculate_scores_entailment(w2mg, wlist1, wlist2, metric='maxdot'):
    assert len(wlist1) == len(wlist2), "Expecting the same length"
    word1_idxs = w2mg.words_to_idxs(wlist1)
    word2_idxs = w2mg.words_to_idxs(wlist2)

    scores = np.zeros((len(word1_idxs)))
    for _i, [w1, w2] in enumerate(zip(word1_idxs, word2_idxs)):
      if metric == 'maxdot':
        scores[_i] = w2mg.maxdot(w1, w2)
      elif metric == 'kl':
        scores[_i] = w2mg.max_negkl(w1, w2)
    return scores
