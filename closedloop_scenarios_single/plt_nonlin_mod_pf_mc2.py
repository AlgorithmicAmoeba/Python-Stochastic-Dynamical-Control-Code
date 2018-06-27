# Plot the Linear Model KF MC results
import matplotlib as mpc
import matplotlib.pyplot as plt
import numpy
import pandas
import src.Results as Results

# mcN = 50
# include("nonlin_mod_pf_lin_mpc_mean_mc.jl")
# include("nonlin_mod_pf_lin_mpc_var_conf_90_mc.jl")

mpc.rc("font", family="serif", serif="Computer Modern", size=8)
mpc.rc("text", usetex=True)
mpc.rc("figure", figsize=(6.0, 3))

mc1 = abs(pandas.read_csv("nonlinmod_pf_mean_mc2.csv", header=None).as_matrix())
mc2 = abs(pandas.read_csv("nonlinmod_pf_var90_mc2.csv", header=None).as_matrix())


rows, cols = mc1.shape  # all will have the same dimension
ts = [x/10 for x in range(800)]

# Now plot 90 % confidence regions!
plt.figure()
plt.subplot(2, 1, 1)  # mean
for k in range(cols):
    plt.plot(ts, mc1[:, k], "k-", linewidth=0.5)

plt.plot(ts, numpy.ones(rows)*0.49, "g-", linewidth=3.0)
plt.ylabel(r"C$_A$ (I)")
plt.locator_params(nbins=4)
plt.xlim(xmin=0)

plt.subplot(2, 1, 2)  # 90%
for k in range(cols):
    plt.plot(ts, mc2[:, k], "k-", linewidth=0.5)

plt.plot(ts, numpy.ones(rows)*0.49, "g-", linewidth=3.0)
plt.ylabel(r"C$_A$ (II)")
plt.locator_params(nbins=4)
plt.xlim(xmin=0)

plt.xlabel("Time [min]")

A = 412
mcerr1 = 0
for k in range(cols):
    mcerr1 += abs(Results.calc_error3(mc1[-100:-1, k], A))

print("The average MC error is:", mcerr1/cols)

mcerr2 = 0
for k in range(cols):
    mcerr2 += abs(Results.calc_error3(mc2[-100:-1, k], A))

print("The average MC error is:", mcerr2/cols)
plt.savefig("/home/ex/Documents/CSC/report/results/Figure_8-28_python.pdf", bbox_inches="tight")
plt.show()
