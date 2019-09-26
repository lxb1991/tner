import logging
import numpy as np


class WordVocab:

    def __init__(self, word_embed_dim, char_embed_dim=0, word_lower=False, char_lower=False):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}

        self.id2char = {}
        self.char2id = {}

        self.word_lower = word_lower
        self.char_lower = char_lower

        self.word_embed_dim = word_embed_dim
        self.word_embeddings = None
        self.char_embed_dim = char_embed_dim
        self.char_embeddings = None

        self.pad_token = '<pad>'
        self.unk_token = '<unk>'

        for token in [self.pad_token, self.unk_token]:
            self.add_word(token)
            self.add_char(token)

    def __len__(self):
        return len(self.token2id)

    def add_word(self, token):

        token = token.lower() if self.word_lower else token
        if token in self.token2id:
            idx = self.token2id[token]
            self.token_cnt[token] += 1
        else:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            self.token_cnt[token] = 1

        return idx

    def add_char(self, char):
        char = char.lower() if self.char_lower else char
        if char in self.char2id:
            idx = self.char2id[char]
        else:
            idx = len(self.char2id)
            self.char2id[char] = idx
            self.id2char[idx] = char
        return idx

    def filter_tokens_by_cnt(self, min_cnt):

        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]

        self.token2id = {}
        self.id2token = {}
        for token in [self.pad_token, self.unk_token]:
            self.add_word(token)
        for token in filtered_tokens:
            self.add_word(token)

    def word_vocab_size(self):
        return len(self.token2id)

    def char_vocab_size(self):
        return len(self.char2id)

    def get_word_id(self, token):
        if token in self.token2id:
            return self.token2id[token]
        else:
            return self.token2id[self.unk_token]

    def get_char_id(self, char):
        if char in self.char2id:
            return self.char2id[char]
        else:
            return self.char2id[self.unk_token]

    def load_word_embeddings(self, embedding_path):
        logger = logging.getLogger("ner")
        oov_word = 0
        glove_word_embeddings = {}
        scale = np.sqrt(3.0 / self.word_embed_dim)
        with open(embedding_path, 'r', encoding="utf-8") as fin:
            for line in fin:
                contents = line.strip().split()
                token = contents[0]
                if len(contents[1:]) != self.word_embed_dim:
                    continue
                glove_word_embeddings[token] = list(map(float, contents[1:]))

        logger.info("glove embeddings size {}".format(len(glove_word_embeddings)))
        self.word_embeddings = np.zeros([self.word_vocab_size(), self.word_embed_dim])
        for token in self.token2id:
            if token in glove_word_embeddings:
                self.word_embeddings[self.get_word_id(token)] = glove_word_embeddings[token]
            elif token.lower() in glove_word_embeddings:
                self.word_embeddings[self.get_word_id(token)] = glove_word_embeddings[token.lower()]
            elif token in [self.pad_token, self.unk_token]:
                self.word_embeddings[self.get_word_id(token)] = np.zeros(self.word_embed_dim)
            else:
                self.word_embeddings[self.get_word_id(token)] = np.random.uniform(-scale, scale, self.word_embed_dim)
                oov_word += 1
        logger.info("oov rate {0}".format(float(oov_word) / self.word_vocab_size()))

    def random_char_embeddings(self):
        self.char_embeddings = np.zeros([self.char_vocab_size(), self.char_embed_dim])
        scale = np.sqrt(3.0 / self.char_embed_dim)
        for index in range(self.char_vocab_size()):
            # if self.id2char[index] in [self.pad_token, self.unk_token]:
            #     continue
            # else:
            #     self.char_embeddings[index, :] = np.random.uniform(-scale, scale, [1, self.char_embed_dim])
            self.char_embeddings[index, :] = np.random.uniform(-scale, scale, [1, self.char_embed_dim])

    def random_word_embeddings(self):
        self.word_embeddings = np.zeros([self.word_vocab_size(), self.word_embed_dim])
        scale = np.sqrt(3.0 / self.word_embed_dim)
        for index in range(self.word_vocab_size()):
            if self.id2token[index] in [self.pad_token, self.unk_token]:
                continue
            else:
                self.word_embeddings[index, :] = np.random.uniform(-scale, scale, [1, self.word_embed_dim])


class LabelVocab:

    def __init__(self):
        self.label2id = {}
        self.id2label = {}

        self.pad_token = '<pad>'
        self.add(self.pad_token)

    def __len__(self):
        return len(self.label2id)

    def add(self, label):
        if label not in self.label2id:
            idx = len(self.label2id)
            self.label2id[label] = idx
            self.id2label[idx] = label

    def get_id(self, label):
        if label in self.label2id:
            return self.label2id[label]

    def get_label(self, idx):
        if idx in self.id2label:
            return self.id2label[idx]
