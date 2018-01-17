# Nonlinear plant controlled with a linear MPC using a KF to estimate the state.
# Conventional deterministic constraints.

import closedloop_scenarios_single.mod_kf_lin_mpc_mean

closedloop_scenarios_single.mod_kf_lin_mpc_mean.main(mcN=1, linear=False)
