from AdoDocument import AdoDocument, separator
import pickle

'''
    Iterable for a corpus of ADO Documents that reads and writes in pickle format
'''


class AdoCorpus:

    def __init__(self, inFile):
        self.file = f'{inFile}.pickle'
        self.handle = None

    def __iter__(self):
        if self.handle is not None:
            self.close()

        self.handle = open(self.file, 'rb')
        return self

    def __next__(self):
        try:
            obj = pickle.load(self.handle)
            return AdoDocument(obj['id'], separator.join(obj['hashtags']),
                           separator.join(obj['tokens']))
        except EOFError:
            self.close
            raise StopIteration()

    def save(self, obj):
        if self.handle is None:
            self.handle = open(self.file, 'wb')

        pickle.dump({'id': obj.id, 'hashtags': obj.hashtags, 'tokens': obj.tokens},
                    self.handle, protocol=pickle.HIGHEST_PROTOCOL)

    def close(self):
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()
            self.handle = None

