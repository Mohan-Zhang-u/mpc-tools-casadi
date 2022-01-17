"""
Unit tests for mpctools.

Note that these tests should test low-level functionality. More complete
example scripts should be included in the root directory and added to the list
in runall.py.
"""

import unittest
import numpy as np
import casadi
from .util import safevertcat

class SymTests(unittest.TestCase):
    """Tests compatibility of various operations with symbolics."""
    def setUp(self):
        self.x0 = casadi.SX.sym("x0")
        self.x1 = casadi.SX.sym("x1")
        self.x = casadi.vertcat(self.x0, self.x1)
    
    def test_listvertcat(self):
        check = safevertcat([self.x0, self.x1])
        self.assertEqual(repr(check), repr(self.x))
        
    def test_arrayvertcat(self):
        check = safevertcat(np.array([self.x0, self.x1]))
        self.assertEqual(repr(check), repr(self.x))

    def test_scalarvertcat(self):
        check = safevertcat(self.x0)
        self.assertEqual(repr(check), repr(self.x0))
        
    def test_literalarrayvertcat(self):
        check = safevertcat(np.array([1, 1]))
        self.assertEqual(repr(check), repr(casadi.IM([1, 1])))
        
    def test_literalscalarvertcat(self):
        with self.assertRaises(TypeError):
            safevertcat(1)

if __name__ == "__main__":
    unittest.main()
