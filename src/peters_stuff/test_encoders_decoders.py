from src.peters_stuff.train_position_predictor import GQNPositionPredictor2
import pickle
import numpy as np


def test_persistence():

    batch_size = 16
    im = (np.random.rand(batch_size, 64, 64, 3)*259.999).astype(np.uint8)
    loc = np.random.rand(batch_size, 2)-.5

    encoder = GQNPositionPredictor2(batch_size=batch_size, image_size=(64, 64), cell_downsample=4, n_maps=32)
    encoder.train(im, loc)
    ser = pickle.dumps(encoder)

    encoder2 = pickle.loads(ser)
    encoder2.train(im, loc)



# def test_graph_saving_idea():





if __name__ == '__main__':
    test_persistence()
