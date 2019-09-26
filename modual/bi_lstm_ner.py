from util import loader
from util.wv_vocab import WordVocab, LabelVocab
from util.wv_data import WordVecData
from torch.utils.data import DataLoader
from model.bilstm import BiLstmSeqLabel
from torch.nn import CrossEntropyLoss
from layer.crf import CRF
from torch import optim
from util import conlleval
import torch
import logging


class BiLstmNER(object):

    def __init__(self, param):
        self.logger = logging.getLogger("ner")

        self.train_dir = "data/domain/news/conll/train.conll"
        self.dev_dir = "data/domain/news/conll/dev.conll"
        self.test_dir = "data/domain/news/conll/test.conll"

        self.param = param
        self.word_vocab = WordVocab(param.word_embed_dim, param.char_embed_dim)
        self.label_vocab = LabelVocab()

    def build_vocab(self, sent_label_list):
        for sent_label in sent_label_list:
            for sent_item in sent_label:
                for word, label in sent_item:
                    self.word_vocab.add_word(word)
                    for char in word:
                        self.word_vocab.add_char(char)
                    self.label_vocab.add(label)

    def build_ner_data(self):
        train_sent_label = loader.load_data_from_conll(self.train_dir)
        dev_sent_label = loader.load_data_from_conll(self.dev_dir)
        test_sent_label = loader.load_data_from_conll(self.test_dir)
        self.logger.info("collect ner data from {0}, {1}, {2}".
                         format(self.train_dir, self.dev_dir, self.test_dir))
        self.logger.info("ner data size {0}, {1}, {2}".
                         format(len(train_sent_label), len(dev_sent_label), len(test_sent_label)))
        self.build_vocab([train_sent_label, dev_sent_label, test_sent_label])
        self.logger.info("word vocab size {0} char vocab size {1} label vocab size {2}".
                         format(len(self.word_vocab), self.word_vocab.char_vocab_size(),
                                len(self.label_vocab)))
        return self.load_data(train_sent_label, dev_sent_label, test_sent_label)

    @staticmethod
    def tuple_item(data_item):
        conll_data = []
        for data_sentence in data_item:
            word_seq = []
            tag_seq = []
            for data_word in data_sentence:
                word_seq.append(data_word[0])
                tag_seq.append(data_word[1])
            if len(word_seq) > 0 and len(tag_seq) > 0:
                conll_data.append((word_seq, tag_seq))
        return conll_data

    def load_data(self, train_list, dev_list, test_list):
        train_data_set = WordVecData(self.tuple_item(train_list), self.word_vocab, self.label_vocab,
                                     self.param.max_sentence_len, self.param.max_word_len)
        train_iter = DataLoader(dataset=train_data_set, batch_size=self.param.batch_size, shuffle=True,
                                num_workers=1, collate_fn=train_data_set.padding)

        dev_data_set = WordVecData(self.tuple_item(dev_list), self.word_vocab, self.label_vocab,
                                   self.param.max_sentence_len, self.param.max_word_len)
        dev_iter = DataLoader(dataset=dev_data_set, batch_size=self.param.batch_size, shuffle=False,
                              num_workers=1, collate_fn=dev_data_set.padding)

        test_data_set = WordVecData(self.tuple_item(test_list), self.word_vocab, self.label_vocab,
                                    self.param.max_sentence_len, self.param.max_word_len)
        test_iter = DataLoader(dataset=test_data_set, batch_size=self.param.batch_size, shuffle=False,
                               num_workers=1, collate_fn=test_data_set.padding)
        return train_iter, dev_iter, test_iter

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate, init_lr):
        lr = init_lr / (1 + decay_rate * epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train(self):
        self.logger.info("run bi-lstm model for conll 2003...")

        train_iter, dev_iter, test_iter = self.build_ner_data()
        self.word_vocab.load_word_embeddings(self.param.embedding_files)
        self.word_vocab.random_char_embeddings()

        proj_size = len(self.label_vocab)
        if self.param.crf:
            proj_size += 2
            self.logger.info("the model will build crf layer...")
        ner_model = BiLstmSeqLabel(self.param, self.word_vocab.word_embeddings,
                                   self.word_vocab.char_embeddings, proj_size)
        crf_layer = CRF(len(self.label_vocab), self.param.gpu_flag)

        if self.param.gpu_flag:
            ner_model = ner_model.cuda()
            crf_layer = crf_layer.cuda()

        model_optimizer = optim.SGD(ner_model.parameters(), lr=self.param.lr,
                                    momentum=self.param.momentum, weight_decay=self.param.l2)
        criterion = CrossEntropyLoss(ignore_index=0, reduction='sum')
        for epoch_idx in range(self.param.epochs):
            self.logger.info("train epoch start: {}".format(epoch_idx))
            model_optimizer = self.lr_decay(model_optimizer, epoch_idx, self.param.lr_decay, self.param.lr)
            ner_model.train()
            total_loss = 0
            batch_num = 0
            for train_batch in train_iter:
                tokens_idx, chars_idx, truth_idx, tokens_mask, sentence_len, _, _ = train_batch
                if self.param.gpu_flag:
                    tokens_idx = tokens_idx.to(self.param.device)
                    chars_idx = chars_idx.to(self.param.device)
                    sentence_len = sentence_len.to(self.param.device)
                    truth_idx = truth_idx.to(self.param.device)
                    tokens_mask = tokens_mask.to(self.param.device)
                token_score, _ = ner_model(tokens_idx, chars_idx, sentence_len)
                if self.param.crf:
                    loss = crf_layer.neg_log_likelihood_loss(token_score, tokens_mask.byte(), truth_idx)
                else:
                    loss = criterion(token_score.view(-1, token_score.shape[-1]), truth_idx.view(-1))
                if self.param.mean_loss:
                    loss = loss / tokens_idx.size(0)
                total_loss += loss.item()
                loss.backward()
                model_optimizer.step()
                ner_model.zero_grad()
                batch_num += 1
            self.logger.info("train down total loss {0} for batch num {1}".format(total_loss, batch_num))

            ner_model.eval()
            self.evaluation(dev_iter, ner_model, crf_layer)
            self.evaluation(test_iter, ner_model, crf_layer)

    def evaluation(self, dev_iter, model, crf_layer):
        golden_label = []
        predict_label = []
        with torch.no_grad():
            for dev_batch in dev_iter:
                tokens_idx, chars_idx, truth_idx, tokens_mask, sentence_len, _, _ = dev_batch
                if self.param.gpu_flag:
                    tokens_idx = tokens_idx.to(self.param.device)
                    chars_idx = chars_idx.to(self.param.device)
                    sentence_len = sentence_len.to(self.param.device)
                    tokens_mask = tokens_mask.to(self.param.device)
                token_score, pred_idx = model(tokens_idx, chars_idx, sentence_len)
                if self.param.crf:
                    _, pred_idx = crf_layer._viterbi_decode(token_score, tokens_mask.byte())
                for p_seq, g_seq, t_seq in zip(pred_idx.cpu().numpy().tolist(), truth_idx.cpu().numpy().tolist(),
                                               tokens_mask.cpu().numpy().tolist()):
                    g_labels = []
                    p_labels = []
                    for p_label_idx, g_label_idx, t_mask in zip(p_seq, g_seq, t_seq):
                        if t_mask == 1:
                            p_label = self.label_vocab.get_label(p_label_idx)
                            g_label = self.label_vocab.get_label(g_label_idx)
                            p_labels.append(p_label)
                            g_labels.append(g_label)
                    golden_label.extend(g_labels)
                    predict_label.extend(p_labels)
        precision, recall, f1 = conlleval.evaluate(golden_label, predict_label, verbose=False)
        self.logger.info("eval precision {0}, recall {1}, f1 {2}.".format(precision, recall, f1))
        return f1
