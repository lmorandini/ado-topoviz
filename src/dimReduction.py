import numpy as np, configargparse, glob, os, sys, pandas as pd
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

sys.path.append('../src')

from logger import logger
from bertopic import BERTopic

def main():
    argp = configargparse.ArgParser()
    argp.add('-c', '--my-config', required=False, is_config_file=True,
             help='config file path')
    argp.add('--model_dir', required=True, type=str, env_var='CORPUS_DIR',
             help='input model file directory')
    argp.add('--model_name', required=True, type=str, env_var='MODEL_NAME',
             help='BERT model name')
    argp.add('--output_dir', required=True, type=str, env_var='OUTPUT_DIR',
             help='directory holding output data')
    argp.add('--output_file', required=True, type=str, env_var='OUTPUT_FILE',
             help='file for output data')

    settings = argp.parse_known_args()[0]

    model_file=f'{settings.model_dir}/{settings.model_name}'
    topic_model = BERTopic.load(model_file)
    logger.warning(f'Read model file {model_file}')

    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    topics = sorted(freq_df.Topic.to_list())

    topic_list = sorted(topics)
    frequencies = [topic_model.topic_sizes[topic] for topic in topic_list]
    words = [" | ".join([word[0] for word in topic_model.get_topic(topic)[:5]]) for topic in topic_list]

    all_topics = sorted(list(topic_model.get_topics().keys()))
    indices = np.array([all_topics.index(topic) for topic in topics])
    embeddings = topic_model.c_tf_idf.toarray()[indices]
    embeddings = MinMaxScaler().fit_transform(embeddings)
    embeddings = UMAP(n_neighbors=2, n_components=2, metric='hellinger').fit_transform(embeddings)

    out_file=f'{settings.output_dir}/{settings.output_file}.topics.csv'
    pd.DataFrame({"x": embeddings[:, 0], "y": embeddings[:, 1],
                       "Topic": topic_list, "Words": words, "Size": frequencies}).to_csv(out_file)
    logger.warning(f'Written {out_file}')

if __name__ == '__main__':
    # WARNING log level is used to avoid gensim printing out very verbose logs at INFO level
    logger.warning(f'Started {__file__}')
    main()
    logger.warning(f'ended')
