# Plot the Linear Model KF MC results
import matplotlib as mpc
import matplotlib.pyplot as plt
import pandas
import numpy
import src.Auxiliary as Auxiliary
import src.Ellipse as Ellipse

# mcN = 500
# include("nonlin_mod_pf_lin_mpc_mean_mc.jl")
# mcN = 700
# include("nonlin_mod_pf_lin_mpc_var_conf_90_mc.jl")

mpc.rc("font", family="serif", serif="Computer Modern", size=8)
mpc.rc("text", usetex=True)
mpc.rc("figure", figsize=(6.0, 3))

mc1 = abs(pandas.read_csv("nonlinmod_kf_mean.csv", header=None).as_matrix())
mc2 = abs(pandas.read_csv("nonlinmod_kf_var90.csv", header=None).as_matrix())
mc3 = abs(pandas.read_csv("nonlinmod_pf_mean.csv", header=None).as_matrix())
mc4 = abs(pandas.read_csv("nonlinmod_pf_var90.csv", header=None).as_matrix())


mc1 = Auxiliary.remove_outliers(mc1, 3)
mc2 = Auxiliary.remove_outliers(mc2, 3)
mc3 = Auxiliary.remove_outliers(mc3, 3)
mc4 = Auxiliary.remove_outliers(mc4, 3)

mmc1 = numpy.mean(mc1, axis=1)
mmc2 = numpy.mean(mc2, axis=1)
mmc3 = numpy.mean(mc3, axis=1)
mmc4 = numpy.mean(mc4, axis=1)

cmc1 = numpy.cov(mc1)
cmc2 = numpy.cov(mc2)
cmc3 = numpy.cov(mc3)
cmc4 = numpy.cov(mc4)

# Now plot 90 % confidence regions!
a = 0.5
xs1, ys1 = Ellipse.ellipse(mmc1, cmc1)
cs1 = plt.fill(xs1, ys1, "m", alpha=a, edgecolor="none")
plt.plot(mmc1[0], mmc1[1], "mo", markersize=10)

xs2, ys2 = Ellipse.ellipse(mmc2, cmc2)
cs2 = plt.fill(xs2, ys2, "r", alpha=a, edgecolor="none")
plt.plot(mmc2[0], mmc2[1], "ro", markersize=10)

xs3, ys3 = Ellipse.ellipse(mmc3, cmc3)
cs3 = plt.fill(xs3, ys3, "g", alpha=a, edgecolor="none")
plt.plot(mmc3[0], mmc3[1], "go", markersize=10)

xs4, ys4 = Ellipse.ellipse(mmc4, cmc4)
cs4 = plt.fill(xs4, ys4, "b", alpha=a, edgecolor="none")
plt.plot(mmc4[0], mmc4[1], "bo", markersize=10)

plt.axis(ymin=0.0, xmin=0.0, ymax=5, xmax=4)
#
# # Magenta = mean
# # Red = 90%
# # Green = 99%
# # Blue = 99.9%
plt.xlabel("Mahalanobis area in violation")
plt.ylabel("Time in violation [min]")
plt.legend(["Expected value constraint KF",
            "Expected value constraint PF",
            r"90$\%$ Chance constraint KF",
            r"90$\%$ Chance constraint PF"],
           loc="lower right")
# plt.savefig("/home/ex/Documents/CSC/report/results/Figure_8-27_python.pdf", bbox_inches="tight")
plt.show()

