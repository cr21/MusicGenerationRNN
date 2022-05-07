from keras.models import Sequential, load_model
from keras.layers import LSTM, TimeDistributed, Dense, Dropout, Activation, Embedding
from Config import Config
import os


def save_params(epoch, model):
    if not os.path.exists(Config.MODEL_DIR):
        os.makedirs(Config.MODEL_DIR)
    model.save_weights(os.path.join(Config.MODEL_DIR, 'weights_{}.h5'.format(epoch)))


def load_weights(model, epoch):
    print(type(model))
    model.load_weights(os.path.join(Config.MODEL_DIR, 'weights_{}.h5'.format(epoch)))


def create_model(batch_size=Config.BATCH_SIZE, seq_len=Config.SEQ_LEN, vocab_size=86):
    model = Sequential()
    # add embedding layer
    #  shape (Vocab_length*512)
    model.add(Embedding(vocab_size, 512, batch_input_shape=(batch_size, seq_len)))
    # add lstm cells
    for i in range(3):
        model.add(LSTM(256, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    return model
