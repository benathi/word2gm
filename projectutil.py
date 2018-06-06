import re
from glob import glob
import os

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
