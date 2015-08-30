import unittest

from models.bnn import BNN

class TestBNN(unittest.TestCase):

    def test_compilation(self):
        clf = BNN([(300,200),(200,100)])
