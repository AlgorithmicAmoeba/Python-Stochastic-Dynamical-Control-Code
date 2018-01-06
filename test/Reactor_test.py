# Reactor tests: run my implementation of RK on the nonlinear model and compare
# the solution to Matlab's solution.

import src.Reactor as Reactor
import pandas
import numpy
import pathlib

state_solutions_path = pathlib.Path("state_solutions.csv")
if state_solutions_path.is_file():
    state_solutions = pandas.read_csv("state_solutions.csv", ).as_matrix()  # read in the ideal answers
else:
    state_solutions = pandas.read_csv("test/state_solutions.csv", ).as_matrix()  # read in the ideal answers

cstr = Reactor.Reactor(V=0.1, R=8.314, CA0=1.0, TA0=310.0, dH=-4.78e4,
                       k0=72.0e9, E=8.314e4, Cp=0.239, rho=1000.0, F=100e-3)

h = 0.001  # time discretisation
tend = 5  # end simulation time
ts = [x/1e3 for x in range(0, 5000)]
N = len(ts)
xs = numpy.zeros([2, N])
initial_states = [0.57, 395]

xs[:, 0] = initial_states
# Loop through the rest of time
for t in range(1, N):
    xs[:, t] = cstr.run_reactor(xs[:, t-1], 0.0, h)  # actual plant


# Run the tests
tol = 0.1
assert abs(state_solutions - xs.T).max() < tol
