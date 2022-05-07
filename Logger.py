from Config import Config
import os


class Logger(object):

    def __init__(self, file):
        if not os.path.exists(Config.LOG_DIR):
            os.makedirs(Config.LOG_DIR)
        self.file = os.path.join(Config.LOG_DIR, file)
        self.epoch = 0
        with open(file, 'w') as f:
            f.write("Epoch,Loss,Accuracy\n")

    def push(self, loss, acc):
        self.epoch += 1
        s = "{},{},{}\n".format(self.epoch, loss, acc)
        with open(self.file, 'a') as f:
            f.write(s)
