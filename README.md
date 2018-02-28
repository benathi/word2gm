# Word2GM (Word to Gaussian Mixture)

This is an implementation of the model in *[Athiwaratkun and Wilson](https://arxiv.org/abs/1704.08424), Multimodal Word Distributions, ACL 2017*.

We represent each word in the dictionary as a Gaussian Mixture distribution and train it using a max-margin objective based on expected likelihood kernel energy function.

The BibTeX entry for the paper is:

```bibtex
@InProceedings{athiwilson2017,
    author = {Ben Athiwaratkun and Andrew Gordon Wilson},
    title = {Multimodal Word Distributions},
    booktitle = {Conference of the Association for Computational Linguistics (ACL)},
    year = {2017}
}
```

## Updates
Feb 27 2018: We updated the code to be compatible with tensorflow 1.0+. Training on large datasets also no longer need tf installation from source. In this version, we provide modified skipgram c ops to handle large dataset training. 

## Dependencies
This code is tested on Tensorflow 1.5.0. The code should be compatible with Tensorflow 1.0 and above. \\

Note:This repository was previously compatible with Tensorflow 0.12 but the support for pre tf1.0 will not be maintained. However, you can access it at this [commit](https://github.com/benathi/word2gm/tree/90a3f50cb66d4f863eed90913bc31dbdd8064fd4).

For plotting, we use [ggplot](https://github.com/yhat/ggplot.git)
```
pip install -U ggplot
# or 
conda install -c conda-forge ggplot
# or
pip install git+https://github.com/yhat/ggplot.git
```

## Training Data
The data used in the paper is the concatenation of *ukWaC* and *WaCkypedia_EN*, both of which can be requested [here](http://wacky.sslmit.unibo.it/doku.php?id=download).

We include a script **get_text8.sh** to download a small dataset **text8** which can be used to train word embeddings. We note that we can observe the polysemies behaviour even on a small dataset such as text8. That is, some word such as 'rock' has one Gaussian component being close to 'jazz', 'pop', 'blue' and another Gaussian component close to 'stone', 'sediment', 'basalt', etc.


## Training

For text8, the training script with the proper hyperparameters are in **train_text8.sh**

For UKWAC+Wackypedia, the training script **train_wac.sh** contains our command to replicate the results.


## Steps
Below are the steps for training and visualization with text8 dataset.

0. Compile C skipgram module for tensorflow training. This generates word2vec_ops.so file which we will use when we import this module in the python code. Note that this version of the code supports training on large datasets without compiling the entire Tensorflow library from source (unlike in the previous version of our code).
```
chmod +x compile_skipgram_ops.sh
./compile_skipgram_ops.sh
```

1. Obtain the dataset and train.
```
bash get_text8.sh
python word2gm_trainer.py --num_mixtures 2 --train_data data/text8 --spherical --embedding_size 50 --epochs_to_train 10 --var_scale 0.05 --save_path modelfiles/t8-2s-e10-v05-lr05d-mc100-ss5-nwout-adg-win10 --learning_rate 0.05  --subsample 1e-5 --adagrad  --min_count 5 --batch_size 128 --max_to_keep 100 --checkpoint_interval 500 --window_size 10
# or simply calling ./train_text8.sh
```
See at the end of page for details on training options.

2. Note that the model will be saved at modelfiles/t8-2s-e10-v05-lr05d-mc100-ss5-nwout-adg-win10. The code to analyze the model and visualize the results is in **Analyze Text8 Model.ipynb**. See model API below.


3. We can visualize the word embeddings itself by executing the following command in iPynb:
```
w2gm_text8_2s.visualize_embeddings()
```
This command prepares the word embeddings to be visualized by Tensorflow's Tensorboard. Once the embeddings are prepared, the visualization can be done by shell command:
```
tensorboard --logdir=modelfiles/t8-2s-e10-v05-lr05d-mc100-ss5-nwout-adg-win10_emb --port=6006
```
Then, navigate the browser to (http://localhost/6006) (or a url of the appropriate machine that has the model) and click at the **Embeddings** tab. Note that the **logdir** folder is the "**original-folder**" + "_emb".

## Visualization
The Tensorboard embeddings visualization tools (please use Firefox or Chrome) allow for nearest neighbors query, in addition to PCA and t-sne visualization. We use the following notation: *x:i* refers to the *i*th mixture component of word 'x'. For instance, querying for 'bank:0' yields 'river:1', 'confluence:0', 'waterway:1' as the nearest neighbors, which means that this component of 'bank' corresponds to river bank. On the other hand, querying for 'bank:1' gives the nearest neighbors 'banking:1', 'banker:0', 'ATM:0', which indicates that this component of 'bank' corresponds to financial bank.


We provide visualization (compatible with Chrome and Firefox) for our models trained on *ukWaC+WaCkypedia* for [K=1](http://35.161.153.223:6001), [K=2](http://35.161.153.223:6002), and [K=3](http://35.161.153.223:6003).


## Trained Model
We provide a trained model for K=2 [here](http://35.161.153.223:6004/w2gm-k2-d50.tar.gz). To analyze the model, see **Analyze Model.ipynb**. The code expects the model to be extracted to directory **modelfiles/w2gm-k2-d50/**.


## Training Options

```
arguments:
  -h, --help            show this help message and exit
  --save_path SAVE_PATH
                        Directory to write the model and training summaries.
                        (required)
  --train_data TRAIN_DATA
                        Training text file. (required)
  --embedding_size EMBEDDING_SIZE
                        The embedding dimension size.
  --epochs_to_train EPOCHS_TO_TRAIN
                        Number of epochs to train. Each epoch processes the
                        training data once completely.
  --learning_rate LEARNING_RATE
                        Initial learning rate.
  --batch_size BATCH_SIZE
                        Number of training examples processed per step (size
                        of a minibatch).
  --concurrent_steps CONCURRENT_STEPS
                        The number of concurrent training steps.
  --window_size WINDOW_SIZE
                        The number of words to predict to the left and right
                        of the target word.
  --min_count MIN_COUNT
                        The minimum number of word occurrences for it to be
                        included in the vocabulary.
  --subsample SUBSAMPLE
                        Subsample threshold for word occurrence. Words that
                        appear with higher frequency will be randomly down-
                        sampled. Set to 0 to disable.
  --statistics_interval STATISTICS_INTERVAL
                        Print statistics every n seconds.
  --summary_interval SUMMARY_INTERVAL
                        Save training summary to file every n seconds (rounded
                        up to statistics interval).
  --checkpoint_interval CHECKPOINT_INTERVAL
                        Checkpoint the model (i.e. save the parameters) every
                        n seconds (rounded up to statistics interval).
  --num_mixtures NUM_MIXTURES
                        Number of mixture component for Mixture of Gaussians
  --spherical [SPHERICAL]
                        Whether the model should be spherical of diagonalThe
                        default is spherical
  --nospherical
  --var_scale VAR_SCALE
                        Variance scale
  --ckpt_all [CKPT_ALL]
                        Keep all checkpoints(Warning: This requires a large
                        amount of disk space).
  --nockpt_all
  --norm_cap NORM_CAP   The upper bound of norm of mean vector
  --lower_sig LOWER_SIG
                        The lower bound for sigma element-wise
  --upper_sig UPPER_SIG
                        The upper bound for sigma element-wise
  --mu_scale MU_SCALE   The average norm will be around mu_scale
  --objective_threshold OBJECTIVE_THRESHOLD
                        The threshold for the objective
  --adagrad [ADAGRAD]   Use Adagrad optimizer instead
  --noadagrad
  --loss_epsilon LOSS_EPSILON
                        epsilon parameter for loss function
  --constant_lr [CONSTANT_LR]
                        Use constant learning rate
  --noconstant_lr
  --wout [WOUT]         Whether we would use a separate wout
  --nowout
  --max_pe [MAX_PE]     Using maximum of partial energy instead of the sum
  --nomax_pe
  --max_to_keep MAX_TO_KEEP
                        The maximum number of checkpoint files to keep
  --normclip [NORMCLIP]
                        Whether to perform norm clipping (very slow)
  --nonormclip

```
