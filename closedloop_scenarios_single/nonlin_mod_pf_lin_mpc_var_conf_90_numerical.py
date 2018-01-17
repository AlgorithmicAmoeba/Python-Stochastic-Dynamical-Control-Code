# Nonlinear plant model controlled with a linear MPC. The control goal is to steer
# the system to the unstead operating point. Stochastic contraints. Numerical evaluation
# of the Gaussian assumption.

import closedloop_scenarios_single.mod_kf_lin_mpc_var_conf

closedloop_scenarios_single.mod_kf_lin_mpc_var_conf.main(nine=90, linear=False, pf=True, numerical=True)
