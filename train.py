# Training Loop
import os
import json
import argparse

import numpy as np
from read_data import read_files
from Config import Config
from create_model import create_model, save_params, load_weights
from Logger import Logger


def read_batches(train_text, vocab_size):
    length = train_text.shape[0];  # 129,665
    num_batches = int(length / Config.BATCH_SIZE);  # 8,104
    for start in range(0, num_batches - Config.SEQ_LEN, Config.SEQ_LEN):  # (0, 8040, 64)

        X = np.zeros((Config.BATCH_SIZE, Config.SEQ_LEN))  # 16X64
        Y = np.zeros((Config.BATCH_SIZE, Config.SEQ_LEN, vocab_size))  # 16X64X86
        for batch_idx in range(0, Config.BATCH_SIZE):  # (0,16)
            for i in range(0, Config.SEQ_LEN):  # (0,64)
                X[batch_idx, i] = train_text[num_batches * batch_idx + start + i]  #
                Y[batch_idx, i, train_text[num_batches * batch_idx + start + i + 1]] = 1
        yield X, Y


def train(seq_data, epochs=Config.EPOCHS, save_freq=10):
    # convert each chars to label
    char_to_index = {ch: idx for (idx, ch) in enumerate(sorted(list(set(seq_data))))}
    print("Total Number of unique chars in dataset {} ".format(len(char_to_index.keys())))
    index_to_char = {V: K for K, V in char_to_index.items()}
    vocab_size = len(index_to_char)
    # get model

    # with open(os.path.join(Config.OUT_DIR, 'char_to_idx.json'), 'w') as f:
    #     json.dump(char_to_index, f)

    model = create_model(batch_size=Config.BATCH_SIZE, seq_len=Config.SEQ_LEN, vocab_size=vocab_size)
    model.summary()
    # loss and optimzer configuration
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # prepare data for traning
    # model can't expect a sequence row text data, pass numeric data using labels for each
    # char we generated earlier

    train_data = np.array([char_to_index[ch] for ch in seq_data], dtype=np.int32)
    print(train_data.shape)
    logger = Logger('training_log.csv')
    for epoch in range(epochs):
        print("\n Epoch {}/{}".format(epoch + 1, epochs))

        losses, acces = [], []

        # iterateover each batch of data
        for idx, (X, Y) in enumerate(read_batches(train_data, vocab_size)):
            #             forward pass
            loss, acc = model.train_on_batch(X, Y)
            print("Batch : {}, Loss {}, acc {}".format(idx + 1, loss, acc))
            losses.append(loss)
            acces.append(acc)
        logger.push(np.average(losses), np.average(acces))

        if (epoch + 1) % save_freq == 0:
            save_params(epoch + 1, model)
            print("Saved Model checkpoints to, weights_{}.h5".format(epoch + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on some Music Note.')
    parser.add_argument('--inputdir', default=Config.DATA_DIR,
                        help='name of directory that contains file to train from')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--freq', type=int, default=10, help='checkpoint save frequency')
    args = parser.parse_args()

    if not os.path.exists(Config.LOG_DIR):
        os.makedirs(Config.LOG_DIR)
    music_notes = read_files(args.inputdir)
    train(music_notes, args.epochs, args.freq)
