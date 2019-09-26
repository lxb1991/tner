from torch.nn import Module, LSTM, Linear, Dropout, Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
import torch.nn.functional as F
import torch


class WordEmbed(Module):

    def __init__(self, word_embedding):
        super(WordEmbed, self).__init__()
        word_embeddings_weight = torch.FloatTensor(word_embedding)
        self.word_matrix = Embedding.from_pretrained(word_embeddings_weight, freeze=False)

        # self.word_matrix = nn.Embedding(word_embedding.shape[0], word_embedding.shape[1])
        # self.word_matrix.weight.data.copy_(torch.from_numpy(word_embedding))

    def forward(self, word_ids):
        return self.word_matrix(word_ids)


class CharacterCNN(Module):

    def __init__(self, char_embedding, char_embed_dim, char_hidden_dim, dropout_rate):
        super(CharacterCNN, self).__init__()
        char_embeddings_weight = torch.FloatTensor(char_embedding)
        self.char_matrix = Embedding.from_pretrained(char_embeddings_weight, freeze=False)

        # self.char_matrix = nn.Embedding(char_embedding.shape[0], char_embedding.shape[1])
        # self.char_matrix.weight.data.copy_(torch.from_numpy(char_embedding))
        self.drop_out = Dropout(p=dropout_rate)
        self.char_cnn_layer = nn.Conv1d(char_embed_dim, char_hidden_dim, kernel_size=3, padding=1)

    def forward(self, char_ids):
        char_embeddings = self.char_matrix(char_ids)
        batch_size = char_embeddings.size(0)
        sent_len = char_embeddings.size(1)
        word_len = char_embeddings.size(2)
        char_dim = char_embeddings.size(3)
        char_embeddings = self.drop_out(char_embeddings).view(-1, word_len, char_dim)
        feature_map = self.char_cnn_layer(char_embeddings.transpose(2, 1))
        char_cnn_out = F.max_pool1d(feature_map, kernel_size=feature_map.size(2)).\
            view(batch_size, sent_len, -1)
        return char_cnn_out


class BiLstm(Module):

    def __init__(self, input_size, lstm_hidden, layer_num):
        super(BiLstm, self).__init__()
        self.lstm = LSTM(input_size, lstm_hidden, num_layers=layer_num, batch_first=True,
                         bidirectional=True)

    def forward(self, word_encode, word_lens):
        pack_word = pack_padded_sequence(word_encode, word_lens, batch_first=True)
        lstm_encode, _ = self.lstm(pack_word)
        lstm_out, _ = pad_packed_sequence(lstm_encode, batch_first=True)
        return lstm_out


class ProjectLayer(Module):

    def __init__(self, input_size, project_size, dropout_rate):
        super(ProjectLayer, self).__init__()
        self.drop_out = Dropout(p=dropout_rate)
        self.project = Linear(input_size, project_size)

    def forward(self, layer_input):
        project_input = self.drop_out(layer_input)
        return self.project(project_input)

