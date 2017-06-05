import re
from glob import glob
import os
import numpy as np

def find_list_ckpts(folder_name, prefix_dir=''):
    # make this callable from any folder
    dir_path = os.path.dirname(os.path.realpath(__file__))
    pattern_txt = os.path.join(prefix_dir, folder_name, "*.ckpt-*")
    filename_list = glob(os.path.join(dir_path, pattern_txt))
    ckpt_list = []
    for fname in filename_list:
        #print fname
        ob = re.search(r'.*.ckpt-(\d+).*', fname)
        if ob:
            #print ob.group(1)
            ckpt_list.append(int(ob.group(1)))
    ckpt_list = set(ckpt_list)
    ckpt_list = list(ckpt_list)
    ckpt_list.sort()

    # now we have a sorted list
    # Next, we reconstruction the model file
    fname_template = os.path.join(dir_path, '{}/{}/model.ckpt-{}')
    ckpt_fnames = [fname_template.format(prefix_dir, folder_name, num) for num in ckpt_list]
    return (ckpt_list, ckpt_fnames)

# options: can change this 
char_emb = np.concatenate((np.zeros((1,26)), np.identity(26)), axis=0)
print char_emb.shape

# takes a numpy
def idxs_to_charseq(id2word, idxs, max_len=10, char_emb_size=26):
  batch_size = idxs.shape[0]
  print 'batch size =', batch_size
  seq = np.zeros((batch_size, max_len, char_emb_size), dtype='int32')
  for i in range(batch_size):
    word = id2word[idxs[i]]
    word_seq = np.array([ord(c) for c in word]) - ord('a') + 1
    print 'shape of charemb_word_seq', char_emb[word_seq].shape
    seq[i,:len(word_seq)] = char_emb[word_seq]
  return seq

# TODO see how to do this for Tensorflow variables
def idxs_to_charseq_tf():
  pass


if __name__ == '__main__':
  result = idxs_to_charseq(['foo', 'bar'], np.array([0,1,0]))
  print result.shape
  print result