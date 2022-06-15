# ADO-TopoViz

Visualization of Dynamic Topic Models as topographic features.


## Overview

- Hashtags can be used as "ground truth" to train models (about only 14% of Tweets have hashtags)
- Accuracy measure baased on hashtags?
- word2vec measure for clustering
- doc2vec (conversation id as document id?) measure
- tf-idf measure
- LDA
- LDA2Vec https://github.com/cemoody/lda2vec
- K-means with measure

[1] An evaluation of document clustering and topic modelling in two online social networks: Twitter and Reddit Stephan
A. Curiskis⁎, Barry Drake, Thomas R. Osborn, Paul J. Kennedy, Information Processing & Management Volume 57, Issue 2,
March 2020
`https://doi.org/10.1016/j.ipm.2019.04.002`
[2] Distributed Representations of Sentences and Documents, Quoc V. Le, Tomas Mikolov,
`https://arxiv.org/search/cs?searchtype=author&query=Mikolov%2C+T`


## Requirements

- Python 3.9.x
- Pip 20.x.x


## Installation

Install dependencies:

```shell
pip install -r requirements.txt
```

Download NTLK ancillary data:

```shell
python -m nltk.downloader -d ./nltk_data stopwords
python -m nltk.downloader -d ./nltk_data punkt
python -m nltk.downloader -d ./nltk_data averaged_perceptron_tagger
python -m gensim.downloader -d ./nltk_data --download glove-twitter-200
```

All the other parameters are set in the `configuration/pipeline-settings.sh` and written, via
the `configuration/pipeline-settings.tpl` to the `configuration/pipeline-settings.conf` file that is used by all the
Python scripts.

In addition, parameters can be passed in the command-line of the various Python scripts.


## Corpus preparation

The initial corpus has to be written as one or more JSON files, following this format:
```json
{"rows":[
  {"key":[2022,2,22,"1001553536138792961","1311306326711001089"],
   "value":{"tags":"","tokens":"tyra|banks|voice|have|ONE|humbler|hand"}},
  {"key":[2022,2,22,"1001553536138792961","1311306326711001089"],
    "value":{"tags":"here","tokens":"There"}},
  {"key":[2022,2,22,"1002011177931620352","1195592725829079040"],
    "value":{"tags":"here|there","tokens":"And|Or"}}
 ]
}

The JSON files should be stored in the directory called `data` at the top level of the repo.
```

Where:
  - `key` is composed of: year, month, day, conversationid, tweetid
  - `value` is composed of: 
     - `tags` a string with all the hashtags separated by '|' and without the hash sign
     - `tokens` a string with all the tweet works separated by '|'

Once the JSON files are ready, they can be processed in a loop with a script like the following:
```shell
  export LOG_LEVEL='WARNING'
  export NLTK_DIR=./nltk_data

  for f in ./data/*.json
  do
    python ./src/buildCorpus.py\
      --corpus_query_file="${f}"\
      --corpora_dir=/tmp\
      --tm_mintokens_perdocument=10\
      --corpus_useconversation=true\
      --corpus_file="$(basename ${f}).corpus"
  done
```
(Explanation of the options can be read with `python ./src/buildCorpus.py --help`.)


## Topic Modelling

Once the Pickle files with the corpus are ready, the topic modelling can be started with a script like the following:
```shell
  export LOG_LEVEL='WARNING'
  export TOKENIZERS_PARALLELISM=true

  python ./src/buildDynTopics.py\
     --corpora_dir='/tmp'\
     --output_dir='/tmp'\
     --corpus_prefix="twitter-2022"\
     --model_name="0-3"\
     --bert_min_topic_size 1000\
     --x_scale 1000\
     --y_scale 1000\
     --sample_fraction=128
```
(Explanation of the options can be read with `python ./src/buildDynTopics.py --help`.)


### Computation of the Surfaces

The finale surface can be computed with a script like the following: 
```shell
  export LOG_LEVEL='WARNING'

  python ./src/computeSurface.py\
      --input_dir='/tmp'\
      --output_dir='/tmp'\
      --model_name='twitter-2022-0-3'\
      --n_rows 600\
      --z_scale 200\
      --max_dist 0.9      
```
(Explanation of the options can be read with `python ./src/computeSurface.py --help`.)
