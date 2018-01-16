# Linear Plant controlled with a linear MPC using a KF to estimate the state.
# Stochastic constraints.

import closedloop_scenarios_single.mod_kf_lin_mpc_var_conf

closedloop_scenarios_single.mod_kf_lin_mpc_var_conf.main(99, mcN=2, linear=True)
