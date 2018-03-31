import unittest
import os
from vagent_mems_ctrnn import VAgent_MEMS_CTRNN


class TestVAgent_MEMS_CTRNN(unittest.TestCase):
    def test_CTRNN_load(self):
        c = VAgent_MEMS_CTRNN()
        c.load('models/sample_vagent.ns')

        self.assertEqual(c.size, 14)
        self.assertEqual(c.mem_h, 2.6e-6)
        self.assertEqual(c.mem_rho, 2330)
        self.assertEqual(c.mem_ythr, -3e-5)
        self.assertEqual(int(c.taus[0]), 1)
        self.assertEqual(int(c.v_biases[0]), -3)
        self.assertEqual(int(c.hs[0]), 2)
        self.assertEqual(c.weights[13][0], 0)

    def test_CTRNN_save(self):
        s_fname = 'tmp.ns'

        c = VAgent_MEMS_CTRNN()
        c.load('models/sample_vagent.ns')
        c.save(s_fname)

        c2 = VAgent_MEMS_CTRNN()
        c2.load(s_fname)

        self.assertEqual(c.size, c2.size)
        self.assertEqual(c.mem_h, c2.mem_h)
        self.assertEqual(c.mem_rho, c2.mem_rho)
        self.assertEqual(c.weights[13][0], c2.weights[13][0])
        self.assertEqual(c.mem_ythr, c2.mem_ythr)

        os.remove(s_fname)

        c2.print_model_abstract()
