import os
import re

from pathlib import Path

import nltk
import numpy as np
from typing import Optional
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer

os.environ['NLTK_DATA'] = str(Path("../datasets/nltk/").resolve())
nltk.data.path.append(str(Path("../datasets/nltk/").resolve()))

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

class MyCountVectorizer():
    def __init__(self,) -> None:
        self.bag_of_words: list = []

    def fit(self, dataset: list[str]) -> None:
        """
        Builds the Bag of Words.

        Args:
            dataset (str): Dataset that contains the main words.
        """
        dataset = " ".join(dataset).lower()
        self.bag_of_words = dict(zip(*self._get_unique_words(dataset, counts = True)))
        self.bag_of_words = self.vocabulary_

    def _get_unique_words(self, data: list[str], counts: Optional[bool] = False) -> np.ndarray | tuple[np.ndarray]:
        words = re.findall(r'\b\w+\b', data.lower())
        return np.unique(words, return_counts = counts)
    
    @property
    def vocabulary_(self) -> dict:
        return {k: v for k, v in sorted(self.bag_of_words.items(), key=lambda item: item[1], reverse = True)}

    def transform(self, data: list[str], normalize: Optional[str] = None) -> list:
        vector = []
        
        for sentence in data:
            unique_words, counts = self._get_unique_words(sentence, counts = True)
            transformed_words = dict(zip(unique_words, counts))
            transformed_counts = [transformed_words.get(word, 0) for word, _ in self.bag_of_words.items()]

            vector.append(transformed_counts)

        matrix = np.array(vector)
        if normalize:
            matrix = self._normalize(matrix, normalize)
        return matrix
    
    def _normalize(self, matrix: np.ndarray, norm: str) -> np.ndarray:
        """Normalize the matrix with L1 or L2 norm."""
        if not norm:
            return matrix 
        
        if norm == 'l1':
            return matrix / np.sum(np.abs(matrix), axis=1, keepdims=True)
        elif norm == 'l2':
            return matrix / np.sqrt(np.sum(matrix**2, axis=1, keepdims=True))
        return matrix
    
    def fit_transform(self, dataset: list[str], normalize: Optional[str] = None):
        self.fit(dataset)
        return self.transform(dataset, normalize)

class LemmaTokenizer():
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    
    def __call__(self, doc: str) -> list:
        tokens = word_tokenize(doc)
        
        words_and_tags = nltk.pos_tag(tokens)
        return [self.wnl.lemmatize(word, get_wordnet_pos(tag)) for word, tag in words_and_tags]

class PorterTokenizer():
    def __init__(self):
        self.porter = PorterStemmer()
    
    def __call__(self, doc: str) -> list:
        tokens = word_tokenize(doc)
        return [self.porter.stem(t) for t in tokens]

class SimpleTokenizer():
    def __init__(self):
        pass

    def __call__(self, doc: str):
        return doc.split()