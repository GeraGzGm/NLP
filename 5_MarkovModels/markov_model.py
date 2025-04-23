import string
import numpy as np
from sklearn.model_selection import train_test_split

class MarkovModel:
    def __init__(self, samples: list, mapper: list):
        self.samples = samples
        self.mapper = mapper

    def train(self, epsilon: float = 1) -> None:
        """
        Get the State Transition Matrix (A) and the Initial State (π)
        """
        self._init_states(len(self.mapper))

        for tokens in self.samples:
            first_token = tokens[0]
            self.pi[first_token] += 1

            for prev_token, current_token in zip(tokens[:-1], tokens[1:]):
                self.A[current_token, prev_token] += 1

        if epsilon > 0:
            self.A += epsilon
            self.pi += epsilon

        #Normalize
        self.A /= self.A.sum(axis = 0, keepdims = True)
        self.pi /= self.pi.sum()

        return self.A, self.pi
    
    def _init_states(self, size: int) -> None:
        self.A = np.zeros((size, size))
        self.pi = np.zeros(size)

class Word2Idx:
    @classmethod
    def map_word2idx(cls, dataset: list) -> dict:
        word2idx = {"<unk>": 0}

        for line in dataset:
            tokens = line.split()
            for token in tokens:
                if token not in word2idx:
                    word2idx[token] = len(word2idx)
        return word2idx
    

    @classmethod
    def map_to_int(cls, dataset: list, mapper: dict) -> list:
        dataset_int = []
        for line in dataset:
            tokens = line.split()
            dataset_int.append([mapper.get(token, 0) for token in tokens])
        return dataset_int


class LoadDataset:
    def __init__(self, text_path: str, label: int):
        self.file = self._load(text_path)
        self.label = label

    def _load(self, path: str):
        with open(path, mode = "r", newline = "\n") as file:
            return file.read().splitlines()
    
    def clean_lines(self) -> list:
        input_text = []

        for line in self.file:
            line = line.rstrip().lower()

            if not line:
                continue

            line = line.translate( str.maketrans("", "", string.punctuation) )
            input_text.append(line)

        return input_text

    def get_samples(self) -> tuple[list, list]:
        input_text = self.clean_lines()
        return input_text, [self.label for _ in range(len(input_text))]
    
    @classmethod
    def get_datasets(cls, x1: list, y1: list, x2: list, y2: list, train_ratio: float) -> tuple[list, list, list, list]:
        x = x1 + x2
        y = y1 + y2

        train = int(train_ratio * len(x))
        return train_test_split(x, y, train_size = train, shuffle = True, random_state = 22)