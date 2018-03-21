import unittest
import math
from utils import sigmoid, inverse_sigmoid
from CTRNN import CTRNN


class TestCTRNN(unittest.TestCase):
    def test_sigmoid(self):
        self.assertEqual(math.floor(sigmoid(12)), 0)
        self.assertEqual(sigmoid(0), 0.5)
        self.assertEqual(math.ceil(sigmoid(12)), 1)

    def test_inverseSigmoid(self):
        self.assertEqual(math.floor(inverse_sigmoid(0.5)), 0)

    def test_CTRNN_load(self):
        c = CTRNN()
        c.load('categorize.ns')

        self.assertEqual(c.size, 14)
        self.assertEqual(int(c.taus[0]), 1)
        self.assertEqual(int(c.biases[0]), -3)
        self.assertEqual(int(c.gains[0]), 2)
        self.assertEqual(c.weights[13][0], 0)
