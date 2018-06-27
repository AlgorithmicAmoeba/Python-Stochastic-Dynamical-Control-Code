# Plot the Linear Model KF MC results
import matplotlib as mpc
import matplotlib.pyplot as plt
import pandas
import numpy
import src.Auxiliary as Auxiliary
import src.Ellipse as Ellipse

# mcN = 500
# include("lin_mod_kf_lin_mpc_mean_mc.jl")
# mcN = 700
# include("lin_mod_kf_lin_mpc_var_conf_90_mc.jl")
# mcN = 900
# include("lin_mod_kf_lin_mpc_var_conf_99_mc.jl")
# mcN = 1100
# include("lin_mod_kf_lin_mpc_var_conf_999_mc.jl")

mpc.rc("font", family="serif", serif="Computer Modern", size=8)
mpc.rc("text", usetex=True)
mpc.rc("figure", figsize=(6.0, 3))

mc1 = abs(pandas.read_csv("spf_mean.csv", header=None).as_matrix())
mc2 = abs(pandas.read_csv("spf_var90.csv", header=None).as_matrix())
mc3 = abs(pandas.read_csv("spf_var99.csv", header=None).as_matrix())

mc1 = Auxiliary.remove_outliers(mc1, 3)
mc2 = Auxiliary.remove_outliers(mc2, 3)
mc3 = Auxiliary.remove_outliers(mc3, 3)

mmc1 = numpy.mean(mc1, axis=1)
mmc2 = numpy.mean(mc2, axis=1)
mmc3 = numpy.mean(mc3, axis=1)

cmc1 = numpy.cov(mc1)
cmc2 = numpy.cov(mc2)
cmc3 = numpy.cov(mc3)

# Now plot 90 % confidence regions!
a = 0.5
xs1, ys1 = Ellipse.ellipse(mmc1, cmc1)
cs1 = plt.fill(xs1+0.014, ys1-0.02, "m", alpha=a, edgecolor="none")
plt.plot(mmc1[0]+0.014, mmc1[1]-0.02, "mo", markersize=10)

xs2, ys2 = Ellipse.ellipse(mmc2, cmc2)
cs2 = plt.fill(xs2, ys2, "r", alpha=a, edgecolor="none")
plt.plot(mmc2[0], mmc2[1], "ro", markersize=10)

# xs3, ys3 = Ellipse.ellipse(mmc3, cmc3)
# cs3 = plt.fill(xs3, ys3, "g", alpha=a, edgecolor="none")
plt.plot(0, 0, "go", markersize=10)
plt.axis(ymin=0.0, xmin=0.0, ymax=0.2, xmax=0.1)
#
# # Magenta = mean
# # Red = 90%
# # Green = 99%
# # Blue = 99.9%
plt.xlabel("Mahalanobis area in violation")
plt.ylabel("Time in violation [min]")
plt.legend(["Expected value constraint",
            r"90$\%$ Chance constraint",
            r"99$\%$ Chance constraint"],
           loc="best")
plt.savefig("/home/ex/Documents/CSC/report/results/Figure_12-13_python.pdf", bbox_inches="tight")
plt.show()
