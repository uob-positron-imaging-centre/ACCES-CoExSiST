# ACCES PARAMETERS START
import coexist
import numpy as np

parameters = coexist.create_parameters(
    ["x", "y"],
    [-np.pi, -np.pi],
    [np.pi, np.pi],
)
# ACCES PARAMETERS END


x, y = parameters["value"]

# Multi-objective optimisation problem taken from:
# http://www.cs.uccs.edu/~jkalita/work/cs571/2012/MultiObjectiveOptimization.pdf

a1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
a2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)

b1 = 0.5 * np.sin(x) - 2 * np.cos(x) + np.sin(y) - 1.5 * np.cos(y)
b2 = 1.5 * np.sin(x) - np.cos(x) + 2 * np.sin(y) - 0.5 * np.cos(y)

f1 = 1 + (a1 - b1)**2 + (a1 - b2)**2
f2 = (x + 3)**2 + (y + 1)**2

error = [-f1, -f2]
