'''
    Load, transform and serialize a corpus (a CouchdDB view JSON) in pickle format.
    The corpus is written as a dictionary, indexed by conversation IDs, with a dictionary as value
    ('hashtags' key for the list of hashtags, and 'tokens' for the list of tweet words)
'''

import sys

sys.path.append('../src')

from logger import logger
from AdoDocument import AdoDocument
from AdoCorpus import AdoCorpus
import json_stream, configargparse

'''
    Retrieve documents from CouchDB, extract terms and return a dictionary 
    indexed by conversation_id 
'''


def retrieveDocuments(corpus_query_file, tm_mintokens_perdocument, corpus_useconversation):
    documents = {}
    try:
        logger.warning(f'Discarding documents with fewer than {tm_mintokens_perdocument} tokens')
        logger.warning(f'Using conversation: {corpus_useconversation}')
        logger.warning(f'Started reading from query file {corpus_query_file}')

        with open(corpus_query_file, 'r') as f:
            rows = json_stream.load(f, persistent=False)['rows']

            for row in rows.persistent():

                k = row['key']
                v = row['value']

                # If the number of tokens in the document is below a threshold, discard the document
                if len(v['tokens']) < tm_mintokens_perdocument:
                    continue

                # If documents has to be a part of a conversation
                if str(corpus_useconversation).lower() == 'true':

                    # If the document is already present in documents, merges its text
                    # with the one already present
                    if documents.get(k[3]):
                        documents[k[3]].add(v['tags'], v['tokens'])
                    else:
                        documents[k[3]] = AdoDocument(k[3], v['tags'], v['tokens'])

                # If the document has to be stored as a single
                else:
                    documents[f'{k[3]}-{k[4]}'] = AdoDocument(f'{k[3]}', v['tags'], v['tokens'])

    except MemoryError:
        logger.error(f'Memory exception')
        sys.exit(1)

    logger.warning(f'started converting tokens to terms')
    {documents[k].tokens2terms() for k in documents}
    return documents


def main():
    argp = configargparse.ArgParser()
    argp.add('--corpus_query_file', required=True, type=str, env_var='CORPUS_QUERY_FILE',
             help='input corpus file name (minus the extension)')
    argp.add('--corpus_file', required=True, type=str, env_var='CORPUS_FILE',
             help='output corpus file')
    argp.add('--corpora_dir', required=False, type=str, env_var='CORPORA_DIR',
             help='directory holding output corpus')
    argp.add('--tm_mintokens_perdocument', required=False, type=int, env_var='TM_MINTOKENS_PERDOCUMENT',
             help='minimum number of tokens to retain a document in the corpus')
    argp.add('--corpus_useconversation', required=False, type=bool, env_var='CORPUS_USECONVERSATION',
             help='flag to use conversations in grouping input documents')

    settings = argp.parse_known_args()[0]

    documents = retrieveDocuments(settings.corpus_query_file, settings.tm_mintokens_perdocument,
                                  settings.corpus_useconversation)

    corpus = AdoCorpus(f'{settings.corpora_dir}/{settings.corpus_file}')
    [corpus.save(v) for v in documents.values()]
    logger.warning(f'Saved corpus to {settings.corpora_dir}/{settings.corpus_file}.pickle')


if __name__ == '__main__':
    # WARNING log level is used to avoid gensim printing out very verbose logs at INFO level
    logger.warning(f'Started {__file__}')
    main()
    logger.warning(f'ended')
