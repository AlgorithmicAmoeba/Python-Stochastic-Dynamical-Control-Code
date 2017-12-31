import matplotlib as mpl
import sys
import os
mpl.use("Agg")


def block_print():
    """Disable print"""
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    """Restore print"""
    sys.stdout = sys.__stdout__


def test_modules():
    modules = ["openloop_scenarios_single.KF_M1",
               "openloop_scenarios_single.KF_M2",
               "openloop_scenarios_single.HMM_Burglar",
               "openloop.Reactor_Compare",
               "openloop.Reactor_Linear_Models",
               "openloop.Reactor_Qualitative"]

    for module in modules:
        print("Testing {0}.py".format(module))
        try:
            block_print()
            exec("import {0}".format(module))
            enable_print()
            print("Test passed\n")
            ans = True
        except ImportError:
            print("Test failed\n")
            ans = False
        assert ans
