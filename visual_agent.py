from CTRNN import CTRNN
from visual_object import Ray
import math


class VisualAgent:

    # diameter of agent
    BODY_SIZE = 30.0
    ENV_WIDTH = 400.0
    MAX_RAY_LENGTH = 220.0
    INPUT_GAIN = 10.0
    VISUAL_ANGLE = math.pi/6
    VEL_GAIN = 5

    def __init__(self, ix=0.0, iy=0.0, num_rays_=7):
        self.cx = 0
        self.cy = 0
        self.vx = 0

        self.num_rays = num_rays_
        self.rays = [Ray() for i in range(self.num_rays)]
        self.nervous_system = CTRNN()
        self.reset(ix, iy)

    # Accessors
    def positionX(self):
        return self.cx

    def set_positionX(self, new_x):
        self.cx = new_x
        self.reset_rays()

    def positionY(self):
        return self.cy

    def reset(self, ix, iy, rs=None, randomize=0):
        self.cx = ix
        self.cy = iy
        self.vx = 0.0
        if randomize:
            self.nervous_system.randomize_circuit_state(-0.1, 0.1, rs)
        else:
            self.nervous_system.randomize_circuit_state(0.0, 0.0, rs)
        self.reset_rays()

    def step(self, step_size, object, show_details=False):
        self.reset_rays()
        for i in range(self.num_rays):
            object.ray_intersection(self.rays[i])
            external_input = VisualAgent.INPUT_GAIN * \
                (VisualAgent.MAX_RAY_LENGTH -
                 self.rays[i].length)/VisualAgent.MAX_RAY_LENGTH
            if show_details is True:
                print('==> ', i, external_input)
                print(self.rays[i])
                print(object)

            self.nervous_system.set_neuron_external_input(i, external_input)

        # Step nervous system
        self.nervous_system.euler_step(step_size)

        # Update agent state
        self.vx = VisualAgent.VEL_GAIN * (self.nervous_system.outputs[12] -
                                          self.nervous_system.outputs[13])

        self.cx += step_size * self.vx

        if self.cx < -VisualAgent.ENV_WIDTH/2:
            self.cx = -VisualAgent.ENV_WIDTH/2
        elif self.cx > VisualAgent.ENV_WIDTH/2:
            self.cx = VisualAgent.ENV_WIDTH/2

    def reset_ray(self, ray, theta, cx, cy):
        if abs(theta) < 0.0000001:
            # special case, vertical ray
            ray.m = math.inf
        else:
            ray.m = 1 / math.tan(theta)

        ray.b = cy - ray.m * cx
        ray.length = VisualAgent.MAX_RAY_LENGTH

        # Set starting coordinates (i.e. on upper perimeter of agent body)
        if ray.m == math.inf:
            ray.startX = cx
            ray.startY = cy + VisualAgent.BODY_SIZE / 2
            return

        ray.startX = cx + (VisualAgent.BODY_SIZE / 2) * math.sin(theta)
        ray.startY = cy + (VisualAgent.BODY_SIZE / 2) * math.cos(theta)

    def reset_rays(self):
        theta = - VisualAgent.VISUAL_ANGLE / 2
        for i in range(self.num_rays):
            self.reset_ray(self.rays[i], theta, self.cx, self.cy)
            theta += VisualAgent.VISUAL_ANGLE/(self.num_rays - 1)
