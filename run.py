import argparse
import random
import torch
import numpy as np
import logging
import os
from modual.bi_lstm_ner import BiLstmNER
from modual.mt_base import MultiBaseNER


def parse_args():
    parser = argparse.ArgumentParser(description="transfer ner model")
    parser.add_argument("--train", action="store_true", help="train the model")
    parser.add_argument("--predict", action="store_true", help="predict the label on test set with trained model")
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")

    model_select = parser.add_argument_group("models")
    model_select.add_argument("--lstm", action="store_true", help="bilstm model")
    model_select.add_argument("--mt_base", action="store_true", help="multi task learning for base ner model")
    model_select.add_argument("--mt_deep", action="store_true", help="multi task learning for deep ner model")
    model_select.add_argument("--mt_adv", action="store_true", help="multi task learning for adversarial ner model")

    model_settings = parser.add_argument_group("model settings")
    model_settings.add_argument("--word_embed_dim", type=int, default=100)
    model_settings.add_argument("--char_embed_dim", type=int, default=30)
    model_settings.add_argument("--char_hidden_dim", type=int, default=50)
    model_settings.add_argument("--lstm_hidden", type=int, default=100)
    model_settings.add_argument("--lstm_layer_num", type=int, default=1)
    model_settings.add_argument("--max_sentence_len", type=int, default=128)
    model_settings.add_argument("--max_word_len", type=int, default=16)
    model_settings.add_argument("--crf", type=bool, default=False)
    model_settings.add_argument("--mean_loss", type=bool, default=False)
    model_settings.add_argument("--drop_out", type=float, default=0.5)

    train_settings = parser.add_argument_group("train settings")
    train_settings.add_argument("--optimizer", type=str, default="SGD")
    train_settings.add_argument("--batch_size", type=int, default=10)
    train_settings.add_argument("--epochs", type=int, default=100)
    train_settings.add_argument("--lr", type=float, default=0.015)
    train_settings.add_argument("--momentum", type=float, default=0)
    train_settings.add_argument("--l2", type=float, default=1e-8)
    train_settings.add_argument("--lr_decay", type=float, default=0.05)

    other_settings = parser.add_argument_group("other settings")
    other_settings.add_argument("--embedding_files", type=str, default="assert/glove.6B.100d.txt")
    return parser.parse_args()


def default_prepare():
    seed_num = 42
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)

    logger = logging.getLogger("ner")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


if __name__ == '__main__':
    logger = default_prepare()
    args = parse_args()

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        args.gpu_flag = True
        args.device = 'cuda'
        logger.info("run model on gpu {}.".format(args.gpu))
    else:
        args.gpu_flag = False
        args.device = "cpu"
        logger.info("run model on cpu.")

    logger.info('Running with args : {}'.format(args))

    our_model = None
    if args.lstm:
        our_model = BiLstmNER(args)
    elif args.mt_base:
        our_model = MultiBaseNER()

    if args.train:
        our_model.train()
    elif args.predict:
        our_model.predict()

