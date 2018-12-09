from ..testutils import BaseTestCase, compare_files, temp_files

import numpy as np
import unittest
import pygsti
import pygsti.construction as pc

class GateConstructionTestCase(BaseTestCase):

    def test_single_qubit_gate_matrix(self):
        expected = np.array([[1.00000000e+00, 2.77555756e-16, -2.28983499e-16, 0.00000000e+00],
                            [ -3.53885261e-16, -8.09667193e-01, 5.22395269e-01, -2.67473774e-01],
                            [ -3.92523115e-17, 5.22395269e-01, 8.49200550e-01, 7.72114534e-02],
                            [ 1.66533454e-16, 2.67473774e-01, -7.72114534e-02, -9.60466643e-01]]
                            )
        mx = pc.single_qubit_gate(24.0, 83.140134, 0.0000)
        self.assertArraysAlmostEqual(expected, mx)

    def test_two_qubit_gate_mx(self):
        gate = pc.two_qubit_gate()
        expected = np.array([
         [ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,],
         [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,]])
        self.assertArraysAlmostEqual(gate,expected)

        
    def test_CNOT_convention(self):
        
        #1-off check (unrelated to fast acton) - showing that CNOT gate convention is CNOT(control,target)
        # so for CNOT:1:2 gates, 1 is the *control* and 2 is the *target*
        from pygsti.construction import std2Q_XYICNOT
        std_cnot = pygsti.tools.process_mx_to_unitary(pygsti.tools.change_basis(std2Q_XYICNOT.target_model.operations['Gcnot'],'pp','std'))
        state_10 = pygsti.tools.dmvec_to_state(pygsti.tools.change_basis(std2Q_XYICNOT.target_model.povms['Mdefault']['10'],"pp","std"))

        # if first qubit is control, CNOT should leave 00 & 01 (first 2 rows/cols) alone:
        expected_cnot = np.array([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                                  [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
                                  [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
                                  [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])

        # first qubit is most significant, so rows are 00,01,10,11
        expected_state10 = np.array([[0.+0.j],
                                     [0.+0.j],
                                     [1.+0.j],
                                     [0.+0.j]]) 
        
        self.assertArraysAlmostEqual(std_cnot, expected_cnot)
        self.assertArraysAlmostEqual(state_10, expected_state10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
