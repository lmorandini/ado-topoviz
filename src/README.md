# ADO-TopicModelling

Analysis of ADO social media posts using Topic Modelling.

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

- Python 3.8.x
- Pip 20.x.x
- Docker 19.3.x (for containerization)
- cURL

## Installation

Install dependencies:

```shell
pip install -r requirements.txt.bak
```

Download NTLK ancillary data:

```shell
python -m nltk.downloader -d ./nltk_data stopwords
python -m nltk.downloader -d ./nltk_data punkt
python -m nltk.downloader -d ./nltk_data averaged_perceptron_tagger
python -m gensim.downloader -d ./nltk_data --download glove-twitter-200
```

Download gensim ancillary data:

```shell
export GENSIM_DATA_DIR="./gensim_data"
python -m gensim.downloader --download glove-twitter-200
```

Create a `secrets.sh` file:

```shell
export COUCHDB_USERNAME="<readonly username>"
export COUCHDB_PASSWORD="<password>"
export COUCHDB_MODELS_USERNAME='<login>'
export COUCHDB_MODELS_PASSWORD='<password>'
```

All the other parameters are set in the `configuration/pipeline-settings.sh` and written, via
the `configuration/pipeline-settings.tpl` to the `configuration/pipeline-settings.conf` file that is used by all the
Python scripts.

In addition, parameters can be passed in the command-line of the various Python scripts.

## Testing

```shell
. ./configuration/pipeline-settings.sh 
cat ./configuration/pipeline-settings.tpl | envsubst > ./configuration/pipeline-settings.conf
export LOG_LEVEL=WARNING
export NLTK_DIR=./nltk_data
python -m unittest unit-tests/corpus.py
python -m unittest unit-tests/similarity.py
python -m unittest unit-tests/word2vec.py
python -m unittest unit-tests/bertTopic.py
#python -m unittest unit-tests/doc2vec.py
python -m unittest unit-tests/mostFrequent.py
```

### Convert cluster metrics to CSV

Emit general metrics for very different number of clusters:

```shell
WORD2VEC_MODEL_FILE="fortnight-20210701-20210730-w2v"
TOPICS='35-35'
FILE="${MODEL_DIR}/${WORD2VEC_MODEL_FILE}-cluster-${TOPICS}"
echo '"ncluster", "silhouettescore", "inertia"' > "${FILE}-general.csv"
jq -r '.clusters[] | [.ncluster, .silhouettescore, .inertia] | @csv' \
  "${FILE}.json" >> "${FILE}-general.csv"
```

Emit individual cluster silhouette metrics and top terms to CSV (it works with the processing of single number of
clusters, not a range):

```shell
WORD2VEC_MODEL_FILE="fortnight-20210701-20210730-w2v"
TOPICS='35-35'
FILE="${MODEL_DIR}/${WORD2VEC_MODEL_FILE}-cluster-${TOPICS}"
echo '"clusterid", "ndocuments", "mean", "min", "max", "terms"' > "${FILE}.csv"
jq -r '.clusters[] | .silhouette[] | [.clusterid, .size, .mean, .min, .max, (.terms | join("|")) ] | @csv'\
   "${FILE}.json" >> "${FILE}.csv" 
```

Open with LibreOffice

```shell
WORD2VEC_MODEL_FILE="fortnight-20210701-20210730-w2v"
TOPICS='35-35'
FILE="${MODEL_DIR}/${WORD2VEC_MODEL_FILE}-cluster-${TOPICS}"
libreoffice "${FILE}.csv" &
```

### Compute cluster's similarity

Compute the closest cluster of two cluster datasets. It requires two input files and one output one:

```shell
export DYNACONF_CLUSTER_A_FILE='fortnight-20210628-20210711-w2v-cluster-35-35'
export DYNACONF_CLUSTER_B_FILE='fortnight-20210712-20210725-w2v-cluster-35-35'
export DYNACONF_SIMILARITY_FILE='fortnight-20210628-20210711-fortnight-20210712-20210725'
python ./src/findSimilarCluster.py
```

Emit the comparison of the aligned clusters, with the top terms of each cluster and the distance between the alignment
to CSV

```shell
export DYNACONF_SIMILARITY_FILE='fortnight-20210628-20210711-fortnight-20210712-20210725'
echo '"distance","clusterA","clusterAterms","clusterB","clusterBterms"' > models/${DYNACONF_SIMILARITY_FILE}.csv
jq -r '.[] | [.distance, .clusterid, (.terms | join("|")), .similar.clusterid, (.similar.terms | join("|"))] | @csv'\
  "${MODEL_DIR}/${DYNACONF_SIMILARITY_FILE}.json" >> "${MODEL_DIR}/${DYNACONF_SIMILARITY_FILE}.csv"
```

Open with LibreOffice

```shell
export DYNACONF_SIMILARITY_FILE='fortnight-20210628-20210711-fortnight-20210712-20210725'
export MODEL_DIR="models"
libreoffice "${MODEL_DIR}/${DYNACONF_SIMILARITY_FILE}.csv" &
```

### Build clusters using LDA

The LDA iterates over hyper-parameters and write them and coherence to `models/coherence.json`:

```shell
python ./src/computeLDA.py 
```

```shell
jq -r '.runs[] | [.ntopics, .coherence] | @csv' models/coherence-singletweets-10-60.json  > models/coherence-singletweets-10-60.csv
```

The program tries out various combination of n. of topics, alpha, and eta parameters and computes a measure of topic
coherence.

```shell
[(0, '0.002*"vaccin" + 0.001*"state" + 0.001*"covid" + 0.001*"amp" + 0.001*"peopl" + 0.001*"case" + 0.001*"health" + 0.001*"death" + 0.001*"school" + 0.001*"govern"'), (1, '0.003*"vaccin" + 0.001*"govern" + 0.001*"covid" + 0.001*"week" + 0.001*"case" + 0.001*"thank" + 0.001*"risk" + 0.001*"death" + 0.001*"viru" + 0.001*"health"')]


MODEL: topics:2 alpha:0.310 eta:0.310
Topic: 0 Word: 0.002*"vaccin" + 0.001*"state" + 0.001*"covid" + 0.001*"amp" + 0.001*"peopl" + 0.001*"case" + 0.001*"health" + 0.001*"death" + 0.001*"school" + 0.001*"govern"
Topic: 1 Word: 0.003*"vaccin" + 0.001*"govern" + 0.001*"covid" + 0.001*"week" + 0.001*"case" + 0.001*"thank" + 0.001*"risk" + 0.001*"death" + 0.001*"viru" + 0.001*"health"
Coherence: 0.439 topics:2 alpha:0.310 eta:0.310


MODEL: topics:5 alpha:0.310 eta:0.310
Topic: 0 Word: 0.002*"vaccin" + 0.001*"case" + 0.001*"state" + 0.001*"peopl" + 0.001*"health" + 0.001*"covid" + 0.001*"risk" + 0.001*"lockdown" + 0.001*"game" + 0.001*"death"
Topic: 1 Word: 0.002*"vaccin" + 0.001*"labor" + 0.001*"women" + 0.001*"govern" + 0.001*"lnp" + 0.001*"parti" + 0.001*"covid" + 0.001*"health" + 0.001*"elect" + 0.001*"case"
Topic: 2 Word: 0.002*"vaccin" + 0.001*"week" + 0.001*"covid" + 0.001*"death" + 0.001*"case" + 0.001*"thank" + 0.001*"govern" + 0.001*"state" + 0.001*"health" + 0.001*"peopl"
Topic: 3 Word: 0.002*"vaccin" + 0.001*"covid" + 0.001*"school" + 0.001*"health" + 0.001*"govern" + 0.001*"case" + 0.001*"state" + 0.001*"mask" + 0.001*"risk" + 0.001*"death"
Topic: 4 Word: 0.002*"vaccin" + 0.001*"game" + 0.001*"govern" + 0.001*"covid" + 0.001*"school" + 0.001*"death" + 0.001*"thank" + 0.001*"week" + 0.001*"day" + 0.001*"lockdown"
Coherence: 0.445 topics:5 alpha:0.310 eta:0.310

```

Topics for hyperparameters with the highest coherence:

```shell
export DYNACONF_LDA_MIN_NTOPICS=5
export DYNACONF_LDA_MAX_NTOPICS=40
export DYNACONF_LDA_STEP_NTOPICS=1
export DYNACONF_LDA_MIN_ALPHA=0.01
export DYNACONF_LDA_MAX_ALPHA=0.01
export DYNACONF_LDA_STEP_ALPHA=0.3
export DYNACONF_LDA_MIN_ETA=0.61
export DYNACONF_LDA_MAX_ETA=0.61
export DYNACONF_LDA_STEP_ETA=0.3
export DYNACONF_COHERENCE_FILE="./models/coherence-singletweets"
export DYNACONF_LDA_MINTOKENS_PERCONVERSATION=10
export DYNACONF_LOG_LEVEL="ERROR"
python ./src/computeLDA.py
```

## Containerisation of pipeline

Docker container for Data processing pipeline for producing word embedding models and clusters for end date and window
size provided.

## Installation

Creation of a shell script with settings (`./configuration/pipeline-settings.sh`)

## Docker image build and push

```shell
. ./configuration/pipeline-settings.sh 
cat ./configuration/pipeline-settings.tpl | envsubst > ./configuration/pipeline-settings.conf 
export DOCKER_SERVER='registry.gitlab.unimelb.edu.au:5005/meg/ado'
docker build --tag ${DOCKER_SERVER}/${APPLICATION_NAME}:${APPLICATION_VERSION} .
```

```shell
. ./configuration/pipeline-settings.sh 
docker push ${DOCKER_SERVER}/${APPLICATION_NAME}:${APPLICATION_VERSION} 
```

## Run the application within a container

Run container with the required env variables, setting the `END_DATE` and `H_WINDOW_SIZE` of the data required in
detached mode.

```shell
. ./configuration/secrets.sh
. ./configuration/pipeline-settings.sh
set -x 
docker rm ${APPLICATION_NAME}-${COUCHDB_DATABASE}
docker run --name ${APPLICATION_NAME}-${COUCHDB_DATABASE}\
  --net=host\
  --env NLTK_DIR="/nltk_data"\
  --env CORPUS_QUERY_PREFIX="/tmp/yearmonthday"\
  --env COUCHDB_SERVER="localhost:5984"\
  --env COUCHDB_DATABASE="${COUCHDB_DATABASE}"\
  --env COUCHDB_USERNAME="${COUCHDB_USERNAME}"\
  --env COUCHDB_PASSWORD="${COUCHDB_PASSWORD}" \
  --env COUCHDB_MODELS_USERNAME="${COUCHDB_MODELS_USERNAME}"\
  --env COUCHDB_MODELS_PASSWORD="${COUCHDB_MODELS_PASSWORD}" \
  --env COUCHDB_MODELS_SERVER="localhost:5984"\
  --env N_WORKERS="${N_WORKERS}"\
  ${DOCKER_SERVER}/${APPLICATION_NAME}:${APPLICATION_VERSION}\
    0 20210705 
set +x    
```

## Passing of END parameter to the Container

To make it work as a CronJob and as a normal Kubernetes Job, the end date (with format) yyyymmdd)
can be passed in different was:

- As the `END` env. var.
- As the second argument to the container (the first is the window size in days)

If both are not given, the day preceding the current date is used.

## Rebase of Topics

Unit tests:

```shell
(
  cd rebase
  export LOG_LEVEL=WARNING
  python -m unittest unit-tests/distances.py
  python -m unittest unit-tests/surfaces.py
)
```

```shell
export LOG_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=true
(
 cd rebase
python ./getTestData.py\
    --my-config ./pipeline-settings.conf\
    --corpora_dir './data'\
    --corpus_file 'newsgroup'\
    --nfolds 10

#export TOKENIZERS_PARALLELISM=false
python ./buildTopics.py\
    --my-config ./pipeline-settings.conf\
    --corpora_dir './data'\
    --models_dir './data'\
    --corpus_name 'newsgroup'\
    --top_n_topics 100\
    --bert_top_n_words 30 \
    --bert_min_topic_size 1000

python ./rebaseTopics.py\
    --my-config ./pipeline-settings.conf\
    --corpora_dir './data'\
    --models_dir './data'\
    --output_dir './viz'\
    --corpus_name 'newsgroup'\
    --model_name 'topic_model'\
    --bert_top_n_words 30\
    --top_n_topics 50\
    --random_state 12345\
    --eps_power 5\
    --x_scale 1000\
    --y_scale 1000\
    --z_scale 0.005\
    --label_terms 5\
    --n_cols 0\
    --n_rows 1000\
    --n_jobs 4\
    --max_cells_dist 20
    
    
python ./src/buildDynTopics.py --my-config="./configuration/pipeline-settings.conf"\
      --corpus_file="${CORPUS_FILE}"
    
)
```

```shell
pip install spacy
python -m spacy download en_core_web_md


(
  cd rebase
  export LOG_LEVEL='WARNING'
  export NLTK_DIR=./nltk_data
  export TOKENIZERS_PARALLELISM=true

  for f in ./data/twitter*
  do
    python ../src/buildCorpustwitter.py --my-config="../configuration/pipeline-settings.conf"\
      --corpora_dir=./data\
      --corpus_query_file="${f}"\
      --corpus_file="$(basename ${f}).corpus"
  done

  python ./buildTopics.py\
    --my-config ./pipeline-settings.conf\
    --corpora_dir './data'\
    --models_dir './data/twitter22'\
    --corpus_name 'twitter'\
    --bert_top_n_words 30 \
    --top_n_topics 50\
    --bert_min_topic_size 10\
    --sample_fraction=16

  python ./rebaseTopics.py\
    --my-config ./pipeline-settings.conf\
    --corpora_dir './data/twitter22'\
    --models_dir './data/twitter22'\
    --output_dir './viz'\
    --corpus_name="twitter"\
    --model_name="twitter"\
    --bert_top_n_words 30\
    --top_n_topics 50\
    --random_state 12345\
    --eps_power 5\
    --x_scale 1000\
    --y_scale 1000\
    --z_scale 0.01\
    --label_terms 5\
    --n_cols 0\
    --n_rows 1000\
    --n_jobs 4\
    --max_cells_dist 20\
    --n_rescaler_reps 100


  python ./buildDynTopics.py\
   --my-config="./pipeline-settings.conf"\
   --corpora_dir='./data'\
   --models_dir='./data'\
   --output_dir='/tmp'\
   --corpus_name="twitter-"\
   --model_name="twitter-dtm"\
   --top_n_topics 0\
   --bert_min_topic_size 1000\
   --x_scale 1000\
   --y_scale 1000\
   --z_scale 60\
   --n_cols 300\
   --n_rows 300\
   --max_cells_dist 60\
   --sample_fraction=1

  python ./rebaseTopics.py\
    --my-config ./pipeline-settings.conf\
    --corpora_dir './data'\
    --models_dir './data'\
    --output_dir './viz'\
    --corpus_name="twitter-"\
    --model_name="twitter-dtm"\
    --bert_top_n_words 30\
    --top_n_topics 100\
    --random_state 12345\
    --eps_power 5\
    --x_scale 1000\
    --y_scale 1000\
    --z_scale 0.01\
    --label_terms 5\
    --n_cols 0\
    --n_rows 1000\
    --n_jobs 4\
    --max_cells_dist 20\
    --n_rescaler_reps 100

)      
```


