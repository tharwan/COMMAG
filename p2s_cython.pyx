#python p2s_setup.py build_ext --inplace

import numpy as np
cimport numpy as np
ctypedef np.int_t DTYPE_t

def updatePhysics(double r_i_sum, double U, double Ri):
	# C = np.sum(r_i)
	cdef double R
	cdef double I

	R = Ri + 1/r_i_sum
	I = U / R
	U_s = U - Ri * I

	return U_s,I,U_s*I