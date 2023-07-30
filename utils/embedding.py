import gzip
import numpy as np

class GloveEmbedding:

    def __init__(self, path):
        self._vocab = []
        self._embeddings = []

        with gzip.open(path, "r") as zf:
            data = zf.read().decode().split("\n")
            data.pop(0)
            data.pop(-1)
            for i, line in enumerate(data):
                line = line.split(" ")
                self._vocab.append(line.pop(0))
                self._embeddings.append(line)
        
        self._embeddings = np.array(self._embeddings, dtype=float)
        

    def __getitem__(self, key):
        try:
            idx = self._vocab.index(key)
            return self._embeddings[idx]
        except:
            return None


class OneHotEmbedding:

    def __init__(self, vocab):
        self._vocab = {char : i for i, char in enumerate(vocab)}
        self._idx2vocab = dict(enumerate(vocab))
        

    def __getitem__(self, key):
        embedding = np.zeros(len(self._vocab))
        embedding[key] = 1.0
        return embedding

    def __call__(self, sentences):
        return np.array([[self.__getitem__(key) for key in sentence] for sentence in sentences])
    
    def decode(self, idxs):
        return [self._idx2vocab[idx] for idx in idxs]
