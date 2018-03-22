import unittest
import math
import os
from utils import sigmoid, inverse_sigmoid
from mems_ctrnn import MEMS_CTRNN


class TestMEMS_CTRNN(unittest.TestCase):
    def test_sigmoid(self):
        self.assertEqual(math.floor(sigmoid(12)), 0)
        self.assertEqual(sigmoid(0), 0.5)
        self.assertEqual(math.ceil(sigmoid(12)), 1)

    def test_inverseSigmoid(self):
        self.assertEqual(math.floor(inverse_sigmoid(0.5)), 0)

    def test_CTRNN_load(self):
        c = MEMS_CTRNN()
        c.load('sample.ns')

        self.assertEqual(c.size, 14)
        self.assertEqual(c.mem_h, 2.6e-6)
        self.assertEqual(c.mem_rho, 2330)
        self.assertEqual(c.mem_ythr, -3e-5)
        self.assertEqual(int(c.taus[0]), 1)
        self.assertEqual(int(c.v_biases[0]), -3)
        self.assertEqual(int(c.hs[0]), 2)
        self.assertEqual(c.weights[13][0], 0)

    def test_CTRNN_save(self):
        s_fname = 'sample2.ns'

        c = MEMS_CTRNN()
        c.load('sample.ns')
        c.save(s_fname)

        c2 = MEMS_CTRNN()
        c2.load(s_fname)

        self.assertEqual(c.size, c2.size)
        self.assertEqual(c.mem_h, c2.mem_h)
        self.assertEqual(c.mem_rho, c2.mem_rho)
        self.assertEqual(c.weights[13][0], c2.weights[13][0])
        self.assertEqual(c.mem_ythr, c2.mem_ythr)

        os.remove(s_fname)
