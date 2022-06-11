import numpy as np, configargparse, glob, os, sys

sys.path.append('../src')

from AdoCorpus import AdoCorpus
from logger import logger
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
from export_lib import exportTopicLocationToGeoJSON
from tuples import TopicLocation

"""
    Build a topic model for a serialized corpus files and serialize the models and 
    related data
"""


def buildTopics(corpus_in, timestamps_in, bert_top_n_words, bert_min_topic_size, sample_fraction):
    logger.warning(f'Corpus size: {len(corpus_in)}')
    corpus = []
    timestamps = []
    for i in range(len(corpus_in)):
        if i % sample_fraction == 0:
            corpus.append(corpus_in[i])
            timestamps.append(timestamps_in[i])
    logger.warning(f'Sampled corpus size: {len(corpus)}')

    topic_model = BERTopic(top_n_words=bert_top_n_words, min_topic_size=bert_min_topic_size)
    (doc_to_topic, doc_to_topic_prob) = topic_model.fit_transform(corpus)
    topics_over_time = topic_model.topics_over_time(corpus, doc_to_topic, timestamps)

    logger.warning('Done computing topics')

    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    topics = sorted(freq_df.Topic.to_list())
    embeddings = np.array(topic_model.topic_embeddings)[
        np.array([sorted(list(topic_model.get_topics().keys())).index(topic) for topic in topics])]
    dist_matrix = 1 - cosine_similarity(embeddings)
    locs = []
    col = 0

    fig = topic_model.visualize_hierarchy()
    leaves = []

    for tt in fig.layout.yaxis.ticktext:
        if len(tt) > 0:
            leaves.append(int(tt.split('_')[0]))
    leaves.reverse()

    ytot = 0.0
    old_leaf = -1
    for leaf in leaves:
        if old_leaf != -1:
            ytot += dist_matrix[old_leaf][leaf]
        old_leaf = leaf

    for t in set(timestamps):
        tt = topics_over_time[topics_over_time['Timestamp'] == t]
        row = 0
        ycum = 0.0
        old_leaf = -1

        for leaf in leaves:

            #  if the topics is "misc", skips
            if leaf == -1:
                continue

            r = tt[tt['Topic'] == leaf]

            # If the topic is empty, skips
            if r.empty == True:
                continue

            if old_leaf != -1:
                ycum += dist_matrix[old_leaf][leaf]

            locs.append(
                TopicLocation(
                    x=col,
                    y=(ycum / ytot) * len(set(timestamps)),
                    t=t,
                    id=leaf,
                    n=r['Frequency'].item(),
                    label=','.join([w[0] for w in topic_model.get_topic(leaf)[0:5]]),
                    top_terms=topic_model.get_topic(leaf)
                )
            )
            row += 1
            old_leaf = leaf
        col += 1

    logger.warning('Done computing locations')
    return locs


def main():
    argp = configargparse.ArgParser()
    argp.add('-c', '--my-config', required=False, is_config_file=True,
             help='config file path')
    argp.add('--corpora_dir', required=False, type=str, env_var='CORPORA_DIR',
             help='directory holding output corpus')
    argp.add('--corpus_prefix', required=True, type=str, env_var='CORPUS_PREFIX',
             help='input corpus file name prefix')
    argp.add('--output_dir', required=True, type=str, env_var='OUTPUT_DIR',
             help='directory holding output data')
    argp.add('--model_name', required=True, type=str, env_var='MODEL_NAME',
             help='output model name')
    argp.add('--bert_top_n_words', required=False, nargs='?', const=1, default=30, type=int, env_var='BERT_TOP_N_WORDS',
             help='number of term returned per cluster')
    argp.add('--bert_min_topic_size', required=False, nargs='?', const=1, default=1000, type=int,
             env_var='BERT_MIN_TOPIC_SIZE',
             help='minimum number of documents to have for a cluster')
    argp.add('--x_scale', required=False, nargs='?', const=1, default=100, type=float, env_var='X_SCALE',
             help='Scale to multiply every unit in the X topic space dimension')
    argp.add('--y_scale', required=False, nargs='?', const=1, default=100, type=float, env_var='Y_SCALE',
             help='Scale to multiply every unit in the y topic space dimension')
    argp.add('--sample_fraction', required=False, type=int, env_var='SAMPLE_FRACTION',
             help='fraction of the corpus to use')

    settings = argp.parse_known_args()[0]
    print(f'{settings.corpora_dir}/{settings.corpus_prefix}*.corpus.pickle')
    corpora_files = glob.glob(f'{settings.corpora_dir}/{settings.corpus_prefix}*.corpus.pickle')
    corpora_files.sort()

    if len(corpora_files) == 0:
       logger.warning(f'No corpus file to read')
       return

    corpus = []
    timestamps = []
    t = 0

    # For every file constituting the corpus
    for corpus_file in corpora_files:
        logger.warning(f'{corpus_file.replace(".pickle", "")}')
        corpus_t = [d.text for d in AdoCorpus(corpus_file.replace('.pickle', ''))]
        corpus += corpus_t
        timestamps += ([t] * len(corpus_t))
        logger.warning(f'Read corpus {corpus_file} {t} {len(corpus)} {len(timestamps)}')
        t += 1

    # Computes topics
    locs = buildTopics(corpus, timestamps,
                       settings.bert_top_n_words, settings.bert_min_topic_size,
                       settings.sample_fraction)

    # Export locations to GeoJSON
    f = f'{settings.output_dir}/{settings.corpus_prefix}-{settings.model_name}'
    exportTopicLocationToGeoJSON(locs, f'{f}.topiclocation.geojson',
                                 settings.x_scale, settings.y_scale)
    logger.warning(f'Written {f}.topiclocation.geojson')


if __name__ == '__main__':
    # WARNING log level is used to avoid gensim printing out very verbose logs at INFO level
    logger.warning(f'Started {__file__}')
    main()
    logger.warning(f'ended')
