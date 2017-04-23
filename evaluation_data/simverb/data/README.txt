This folder contains the accompanying files for the paper: "SimVerb-3500: A Large-Scale Evaluation Set of Verb Similarity", EMNLP 2016 submission.

##############################################################################################################
# Files

 - SimVerb-3500.txt : The main dataset (all 3500 pairs)

We provide a standardised split into development and test set:
 - SimVerb-500-dev.txt : The development set (500 pairs)
 - SimVerb-3000-test.txt : The test set (3000 pairs)

As the main dataset only contains the averaged human ratings, we additionally provide two files with the original ratings:
 - SimVerb-3500-ratings.csv : Original ratings of all pairs. Note this file does not indicate which annotator rated which pairs. 

 - SimVerb-3500-stats.txt: Frequency statistics from the BNC corpus


In the following we describe the format of each file.

###############################################################################################################
# SimVerb-3500.txt, SimVerb-500-dev.txt, SimVerb-3000-test.txt 

The dataset comes in form of a tab separated plain text file, and is compatible with the format used in SimLex-999  (SimLex-999: Evaluating Semantic Models with (Genuine) Similarity Estimation. 2015. Felix Hill, Roi Reichart and Anna Korhonen.) All three versions of the dataset are in the same format.

# word1: The first verb of the pair.

# word2: The second verb of the pair.

# POS: The part-of-speech tag. Note that it is 'V' for all pairs, since the dataset exclusively contains verbs. We decided to include it nevertheless to make it compatible with SimLex-999.

# score: The SimVerb-3500 similarity rating. Note that average annotator scores have been linearly mapped from the range [0,6] to the range [0,10] to match other datasets.

# relation: the lexical relation of the pair. Possible values: "SYNONYMS", "ANTONYMS", "HYPER/HYPONYMS", "COHYPONYMS", "NONE".

###############################################################################################################
# SimVerb-3500-ratings.csv

This file contains all ratings by pair. Note that it does not indicate which annotator rated which pairs.

# word1: The first verb of the pair.

# word2: The second verb of the pair.

# r1-10: Ratings in the range [0,6]. Every pair has at least 10 ratings, some pairs have more ratings.

###############################################################################################################
# SimVerb-3520-annotator-ratings.csv

This file contains a matrix of ratings per annotator. Rows are pairs, columns are annotators.
Ratings are '-1' in case an annotator has not rated a pair, and in the range [0,6] otherwise.
Note this file contains 3520 verb pairs - the additional 20 pairs are the consistency set, rated by all participants.

###############################################################################################################
# SimVerb-3500-stats.txt

This file contains additional statistics per verb lemma regarding frequency (extracted from BNC) and VerbNet class membership.

# COUNTER: Numbers the lemma in the file, goes from 1-827; all verb lemmas were sorted alphabetically.

# VBLEMMA: Verb lemma.

# VBCLASS: The list of classes (delimited by ;) to which the particular verb lemma belongs (can be more than 1, N/A if they don't belong to any class). There are 101 Levin-style classes as in VerbNet.

# BNCFREQ: Absolute frequency in the BNC corpus.

# BNCRNKW: Absolute ranking of verb lemma given all lemmas and their frequencies from the BNC corpus.

# BNCRNKV: Absolute ranking of verb lemma given only verb lemmas and their frequencies from the BNC corpus.

# SIMRNKV: Relative ranking of verb lemma using only the set of 827 verb lemmas from our pool of verb types (values: 1-827).

# BNCQRTV: The quartile to which the verb lemma belongs given the set of all verb lemmas and their frequencies from the BNC corpus.








