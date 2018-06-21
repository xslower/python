from nn import *
import pickle


def load_data():
    fpx = open('data/train_x', mode='rb')
    fpy = open('data/train_y', mode='rb')
    X = pickle.load(fpx)
    Y = pickle.load(fpy)
    fpx.close()
    fpy.close()
    return X, Y


def train():
    X, Y = load_data()
    model = TitleRecKr(100, 22, 300)
    model.train(X, Y)

if __name__ == '__main__':
    train()