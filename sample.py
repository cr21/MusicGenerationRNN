import argparse
import os
import json

import numpy as np

from create_model import create_model, save_params, load_weights
from Config import Config
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding

DATA_DIR = Config.DATA_DIR
MODEL_DIR = Config.MODEL_DIR
OUT_DIR = Config.OUT_DIR


def build_sample_model(vocab_size):
    print(vocab_size)
    model = Sequential()
    model.add(Embedding(vocab_size, 512, batch_input_shape=(1, 1)))
    for i in range(3):
        model.add(LSTM(256, return_sequences=(i != 2), stateful=True))
        model.add(Dropout(0.2))

    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    return model


def sample(epoch, header, num_chars):
    with open(os.path.join(OUT_DIR, 'char_to_idx.json')) as f:
        char_to_idx = json.load(f)
    idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
    vocab_size = len(char_to_idx)
    vocab_size = 86
    model = build_sample_model(vocab_size)

    load_weights(model, epoch)
    model.save(os.path.join(MODEL_DIR, 'model.{}.h5'.format(epoch)))

    sampled = [char_to_idx[c] for c in header]
    print("sampled Char : {} and index : {}".format(header, sampled))

    for i in range(num_chars):
        # create batch of size 1 and seq length 1
        batch = np.zeros((1, 1))
        if sampled:
            batch[0, 0] = sampled[-1]
        else:
            # choose randomly
            batch[0, 0] = np.random.randint(vocab_size)
        result = model.predict_on_batch(batch).ravel()
        # choose randomly with probability returned by model softmax layer
        sample_out = np.random.choice(range(vocab_size), p=result)
        sampled.append(sample_out)

    # convert index back to char
    with open(os.path.join(OUT_DIR, 'generated_music.txt'), 'w') as f:
        f.write(''.join(idx_to_char[c] for c in sampled))
    return ''.join(idx_to_char[c] for c in sampled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample some notes from the trained model.')
    parser.add_argument('--epoch', type=int, help='epoch checkpoint to sample from')
    parser.add_argument('--seed', default='', help='initial seed for the generated text')
    parser.add_argument('--len', type=int, default=512, help='number of characters to sample (default 512)')
    args = parser.parse_args()

    print(sample(args.epoch, args.seed, args.len))
