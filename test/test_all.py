import test.HMM_test as HMM_test
import test.LLDS_test as LLDS_test
import test.PF_test as PF_test
import test.Reactor_test as Reactor_test

HMM_test.filter_test()
HMM_test.smooth_test()
HMM_test.viterbi_test()
HMM_test.prediction_test()

LLDS_test.filter_test()
LLDS_test.smooth_test()

PF_test.filter_test()

Reactor_test.simulation_test()
