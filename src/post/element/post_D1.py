import os
import sys

SRC_DIR = os.path.abspath(r"..\..")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils import distance


class PostRod(object):

    NNODE = 2
    """
    Number of nodes which define the element.
    
    dtype: int
    """

    def __init__(self, rod, post_nodes):
        self._id = rod.id
        self._E = rod.E
        self._length_rigid = rod.length

        if not all(rod.nodes[i].id == post_nodes[i].id for i in range(2)):
            raise ValueError("Incorrect PostNodes were provided to PostRod %d" % self.id)

        self._nodes = post_nodes
        self._length_flex = distance(post_nodes[0].x_flex, post_nodes[1].x_flex)

    @property
    def E(self):
        return self._E

    @property
    def id(self):
        return self._id

    @property
    def length_flex(self):
        return self._length_flex

    @property
    def length_rigid(self):
        return self._length_rigid

    @property
    def nodes(self):
        return self._nodes

    @property
    def strain(self):
        return (self.length_flex - self.length_rigid) / self.length_rigid

    @property
    def stress(self):
        return self.E * self.strain
