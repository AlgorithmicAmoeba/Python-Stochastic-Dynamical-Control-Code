# Linear Plant controlled with a linear MPC using a KF to estimate the state.
# Stochastic constraints.

import closedloop_scenarios_single.lin_mpc_mean

closedloop_scenarios_single.lin_mpc_mean.main(mcN=200, linear=False, pf=False)
