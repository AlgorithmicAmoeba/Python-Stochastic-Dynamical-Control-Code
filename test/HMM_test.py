# Hidden Markov Model
# h - h -> ...
# |   |
# o   o
# Tests for the HMM code.
# All tests are compared to code supplied by:
# Bayesian Reasoning and Machine Learning
# by David Barber
# Website: http://www0.cs.ucl.ac.uk/staff/d.barber/brml/
# The example is taken from Exercise 23.3
# The functions were all compared to their Matlab equivalent.
# I used: HMMforward, HMMsmooth and HMMviterbi

import src.HMM as HMM
import pandas
import numpy
import pathlib

smooth_path = pathlib.Path("smooth_hmm.csv")
if smooth_path.is_file():
    fbs_barber = pandas.read_csv("smooth_hmm.csv", header=None).as_matrix()  # read in the ideal answers
    filter_barber = pandas.read_csv("filter_hmm.csv", header=None).as_matrix()  # read in the ideal answers
else:
    fbs_barber = pandas.read_csv("test/smooth_hmm.csv", header=None).as_matrix()  # read in the ideal answers
    filter_barber = pandas.read_csv("test/filter_hmm.csv", header=None).as_matrix()  # read in the ideal answers

# Discrete model
A = numpy.array([[0.5, 0.0, 0.0], [0.3, 0.6, 0.0], [0.2, 0.4, 1.0]])  # transition probabilities
B = numpy.array([[0.7, 0.4, 0.8], [0.3, 0.6, 0.2]])  # emission probabilities

mod1 = HMM.HMM(A, B)  # create the HMM object
initial = numpy.array([0.9, 0.1, 0.0])  # initial state distribution
evidence = numpy.array([0, 0, 1, 0, 1, 0, 1])  # evidence/observations

filter_me = mod1.forward(initial, evidence)

fbs_me = numpy.zeros([len(initial), len(evidence)])
for k in range(len(evidence)):
    fbs_me[:, k] = mod1.smooth(initial, evidence, k)  # works!

vtb_me = mod1.viterbi_dp(initial, evidence)  # works!
vtb_barber = [0, 2, 2, 2, 2, 2, 2]  # Barber's answer

p_states, p_evidence = mod1.prediction(initial, evidence)  # No test for this - not implemented by barber

# Run the tests
# Viterbi Inference
assert numpy.array_equal(vtb_me, vtb_barber)

# Filter Inference
assert abs(filter_me - filter_barber).max() < 1e-4

# Smoother Inference
assert abs(fbs_me - fbs_barber).max() < 1e-4

