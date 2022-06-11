from collections import namedtuple

TopicData = namedtuple('TopicData', ['model_id', 'topic_embeddings', 'top_terms'])
SimilarityData = namedtuple('SimilarityData', ['model_ids', 'word_similarity', 'cosine_similarity'])
SimilarityList = namedtuple('SimilarityLIst', ['ids', 'similarity'])
TopicLocation = namedtuple('TopicLocation', ['x', 'y', 't', 'id', 'label', 'top_terms', 'n'])
