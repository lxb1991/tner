from torch.utils import data
import torch


class WordVecData(data.Dataset):

    def __init__(self, sent_label_tuple, word_vocab, label_vocab, max_sentence_len, max_word_len):
        self.sent_label_data = sent_label_tuple
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab

        self.max_sentence_len = max_sentence_len
        self.max_word_len = max_word_len

    def __len__(self):
        return len(self.sent_label_data)

    def __getitem__(self, item_idx):
        data_item = self.sent_label_data[item_idx]
        data_word = data_item[0]
        data_tag = data_item[1]

        # raw information
        sentence_text = []
        sentence_tags = []

        # format information
        tokens_idx = []
        chars_idx = []
        tags_idx = []
        tokens_mask = []

        for word, tag in zip(data_word, data_tag):

            token_idx = self.word_vocab.get_word_id(word)
            tag_idx = self.label_vocab.get_id(tag)
            char_idx = []
            for char in word:
                char_idx.append(self.word_vocab.get_char_id(char))
            char_idx = char_idx if len(char_idx) <= self.max_word_len else char_idx[:self.max_word_len]
            chars_idx.append(char_idx)
            tokens_idx.append(token_idx)
            tags_idx.append(tag_idx)
            tokens_mask.append(1)
            sentence_text.append(word)
            sentence_tags.append(tag)
        sentence_len = len(tokens_idx)
        return tokens_idx, chars_idx, tags_idx, tokens_mask, sentence_len, sentence_text, sentence_tags

    def padding(self, batch):

        def batchify_fn(batch_idx):
            return [sample[batch_idx] for sample in batch]

        def pad_fn(batch_idx, max_length):
            return [sample[batch_idx] + [0] * (max_length - len(sample[batch_idx])) for sample in batch]

        def pad_char(max_length):
            return [[c + [0] * (self.max_word_len - len(c)) for c in sample[1]] +
                    [[0] * self.max_word_len] * (max_length - len(sample[1])) for sample in batch]

        batch = sorted(batch, key=lambda x: x[4], reverse=True)
        max_len = max(batchify_fn(4))

        tokens_idx = torch.LongTensor(pad_fn(0, max_len))
        chars_idx = torch.LongTensor(pad_char(max_len))
        tags_idx = torch.LongTensor(pad_fn(2, max_len))

        tokens_mask = torch.LongTensor(pad_fn(3, max_len))
        tokens_len = torch.LongTensor(batchify_fn(4))

        return tokens_idx, chars_idx, tags_idx, tokens_mask, tokens_len, batchify_fn(5), batchify_fn(6)
