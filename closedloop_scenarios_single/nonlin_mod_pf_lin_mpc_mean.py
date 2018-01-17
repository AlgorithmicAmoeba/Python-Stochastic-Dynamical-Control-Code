# Nonlinear plant model controlled with a linear MPC. The control goal is to steer
# the system to the unstead operating point. Deterministic contraints.

import closedloop_scenarios_single.lin_mpc_mean

closedloop_scenarios_single.lin_mpc_mean.main(mcN=1, linear=False, pf=True)
