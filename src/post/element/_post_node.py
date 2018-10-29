import numpy as np


class PostNode(object):

    def __init__(self, node, dx, scale=1.):
        self._id = node.id
        self._ndof = node.ndof
        self._x_rigid = node.x

        if len(dx) != self.ndof:
            raise ValueError("Incorrect number of displacements provided to PostNode %d" % self.id)

        self._dx = np.array(dx)
        self.scale = scale

    @property
    def dx(self):
        return self._dx * self.scale

    @property
    def id(self):
        return self._id

    @property
    def ndof(self):
        return self._ndof

    @property
    def x_flex(self):
        return self.x_rigid + self.dx

    @property
    def x_rigid(self):
        return self._x_rigid
