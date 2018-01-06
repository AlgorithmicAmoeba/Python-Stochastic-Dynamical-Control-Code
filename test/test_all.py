import test.HMM_test as HMM_test
import test.LLDS_test as LLDS_test
import test.PF_test as PF_test
import test.Reactor_test as Reactor_test


def test_all():
    HMM_test.test_filter()
    HMM_test.test_smooth()
    HMM_test.test_viterbi()
    HMM_test.test_prediction()

    LLDS_test.test_filter()
    LLDS_test.test_smooth()

    PF_test.test_filter()

    Reactor_test.test_simulation()


if __name__ == '__main':
    test_all()
