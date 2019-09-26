from torch.nn import Module, Dropout
from layer.common import WordEmbed, CharacterCNN, BiLstm, ProjectLayer
import torch


class BiLstmSeqLabel(Module):

    def __init__(self, param, word_embedding, char_embedding, label_size):
        super(BiLstmSeqLabel, self).__init__()

        self.word_encode_layer = WordEmbed(word_embedding)
        self.char_encode_layer = CharacterCNN(char_embedding, param.char_embed_dim,
                                              param.char_hidden_dim, param.drop_out)
        self.dropout_layer = Dropout(p=param.drop_out)
        bilstm_input_size = param.word_embed_dim + param.char_hidden_dim
        self.bilstm_encode_layer = BiLstm(bilstm_input_size, param.lstm_hidden, param.lstm_layer_num)
        self.linear_decode_layer = ProjectLayer(param.lstm_hidden * 2, label_size, param.drop_out)

    def forward(self, word_idx, char_idx, word_lens):
        word_emb = self.word_encode_layer(word_idx)
        char_emb = self.char_encode_layer(char_idx)
        fusion_token_emb = self.dropout_layer(torch.cat((word_emb, char_emb), -1))
        context_emb = self.bilstm_encode_layer(fusion_token_emb, word_lens)
        token_score = self.linear_decode_layer(context_emb)
        predict_label_ids = torch.argmax(token_score, -1)
        return token_score, predict_label_ids
