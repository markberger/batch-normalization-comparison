import unittest

from models.mlp import MLP

class TestMLP(unittest.TestCase):

    def test_compilation(self):
        clf = MLP([(300,200),(200,100)])
