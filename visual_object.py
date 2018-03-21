import math


# A visual ray – line segment of form y = mx + b with start
# coordinates and length
class Ray:
    def __init__(self, b=0, m=0, startX=0, startY=0, length=0):
        self.b = b
        self.m = m
        self.startX = startX
        self.startY = startY
        self.length = length

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Ray, " + \
               "b={}, m={}, startX={}".format(self.b, self.m, self.startX) + \
               ", startY={}, length={}".format(self.startY, self.length)


# The VisualObject class declaration
class VisualObject:

    # The constructor
    def __init__(self, cx=0.0, cy=275.0, vy=-3.0, size=30.0):
        self.cx = cx
        self.cy = cy
        self.vy = vy
        self.size = size

    # Updates ray length if an intersection occurs
    # Assumes agent is 'looking up' at object
    def ray_intersection(self, ray):
        pass

    # Accessors
    def set_positionX(self, x):
        self.cx = x

    def set_positionY(self, y):
        self.cy = y

    def positionX(self):
        return self.cx

    def positionY(self):
        return self.cy

    def step(self, step_size):
        self.cy += step_size * self.vy

    def who_am_i(self):
        return "VisualObject"

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.who_am_i() + ", "\
               "cx={}, cy={}, vy={}".format(self.cx, self.cy, self.vy) + \
               ", size={}".format(self.size)


# Class for horizontal line segment
class Line (VisualObject):
    # he constructor
    # size --> length of segment
    def __init__(self, cx=0.0, cy=275.0, vy=-3.0, size=30.0):
        super().__init__(cx, cy, vy, size)

    def ray_intersection(self, ray):
        # x_intersect
        if ray.m == math.inf:
            x_intersect = ray.startX
        else:
            x_intersect = (self.cy - ray.b)/ray.m

        # No intersection
        if math.fabs(self.cx-x_intersect) > self.size/2.0 or self.cy < 0:
            return None

        # Intersection, return distance
        new_length = math.sqrt(pow(x_intersect-ray.startX, 2.0) +
                               pow(self.cy-ray.startY, 2.0))
        if new_length < ray.length:
            ray.length = new_length
        return ray

    def who_am_i(self):
        return "Line"


# Class definition for a circle
class Circle (VisualObject):
    # The constructor
    # size --> diameter
    def __init__(self, cx=0.0, cy=275.0, vy=-3.0, size=30.0):
        super().__init__(cx, cy, vy, size)

    def ray_intersection(self, ray):
        # Special case, vertical ray
        if ray.m == math.inf:
            if math.fabs(ray.startX - self.cx) > self.size/2:
                return None

            x_intersect = ray.startX
            A = 1
            B = -2 * self.cy
            C = pow(self.cy, 2) - pow(self.size/2, 2) + \
                pow(x_intersect - self.cx, 2)
            disc = pow(B, 2) - 4 * A * C

            # assuming cy>0 and cy>ray.startY
            y_intersect = (-B - math.sqrt(disc)) / (2 * A)

            new_length = math.fabs(y_intersect - ray.startY)
            if new_length < ray.length:
                ray.length = new_length

            return None

        A = 1 + pow(ray.m, 2)
        B = 2*(ray.m*(ray.b-self.cy)-self.cx)
        C = pow(self.cx, 2) + pow(ray.b, 2) - 2 * ray.b * self.cy + \
            pow(self.cy, 2)-pow(self.size/2, 2)
        disc = pow(B, 2) - 4 * A * C

        if disc < 0:
            return None

        # Find points of intersection, return smaller
        # distance (distances are the same for tangent lines)
        x_intersect1 = (-B + math.sqrt(disc))/(2 * A)
        y_intersect1 = ray.m * x_intersect1 + ray.b
        distance1 = math.sqrt(pow(x_intersect1-ray.startX, 2) +
                              pow(y_intersect1-ray.startY, 2))

        x_intersect2 = (-B - math.sqrt(disc)) / (2 * A)
        y_intersect2 = ray.m * x_intersect2 + ray.b
        distance2 = math.sqrt(pow(x_intersect2-ray.startX, 2) +
                              pow(y_intersect2-ray.startY, 2))

        if distance1 < distance2:
            new_length = distance1
        else:
            new_length = distance2

        if new_length < ray.length:
            ray.length = new_length

        return ray

    def who_am_i(self):
        return "Circle"
