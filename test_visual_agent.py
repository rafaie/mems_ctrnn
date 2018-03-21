import unittest
from visual_object import Ray, Line, Circle
from visual_agent import VisualAgent

# import math
# from utils import sigmoid, inverse_sigmoid
# from CTRNN import CTRNN


class TestVisualAgent(unittest.TestCase):
    def calc_external_input(self, r):
        return VisualAgent.INPUT_GAIN * \
            (VisualAgent.MAX_RAY_LENGTH - r.length)/VisualAgent.MAX_RAY_LENGTH

    def test_intersection(self):
        r = Ray(b=-111.962, m=-3.73205, startX=-33.8823, startY=14.4889,
                length=220)
        l = Line(cx=30, cy=150, vy=-3, size=30)
        l.ray_intersection(r)

        external_input = self.calc_external_input(r)
        self.assertEqual(external_input, 0)

        r = Ray(b=-111.962, m=-3.73205, startX=-33.8823, startY=14.4889,
                length=220)
        c = Circle(cx=30, cy=150, vy=-3, size=30)
        c.ray_intersection(r)

        external_input = self.calc_external_input(r)
        self.assertEqual(external_input, 0)
        # print(' ==> ', external_input)
