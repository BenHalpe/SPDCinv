from SHGCWE import *

crystal_coef = {"max_mode1": 1, "max_mode2":0, "real_coef":np.array([1]),"img_coef":np.array([0])}
A = SHGCWE(check_sol = True,crystal_coef=crystal_coef, draw_sol = True,N=1)
A.solve()