
import scipy.stats
import numpy
import matplotlib as mpc
import matplotlib.pyplot as plt


print("Auxiliary is hardcoded for the CSTR!")


def kl(part_states, part_weights, m, S, temp_states):
    """Discrete Kullback-Leibler divergence test wrt a multivariate normal model."""

    sweights = 1.0 - sum(part_weights)
    N = len(part_weights)
    if sweights < 0.0:  # add some robustness here (very crude)
        part_weights += (1/N)*sweights
        print("Particle weights adjusted by ", sweights, " in Auxiliary!")
    elif sweights > 0.0:
        part_weights += (1 / N) * sweights
        print("Particle weights adjusted by ", sweights, " in Auxiliary!")

    dnorm = scipy.stats.multivariate_normal(mean=m, cov=S)

    for k in range(N):
        j = numpy.random.choice(range(len(part_weights)), size=1, p=part_weights)
        temp_states[:, k] = part_states[:, j]

    estden = scipy.stats.gaussian_kde(temp_states)

    kldiv = 0.0
    for k in range(N):
        # draw from samplesw
        kldiv += -dnorm.logpdf(temp_states[:, k]) + estden.logpdf([temp_states[1, k], temp_states[2, k]])

    return (1.0/N)*kldiv


def klbase(m, S, temp_states, N):

    dnorm = scipy.stats.multivariate_normal(mean=m, cov=S)

    for k in range(N):
        temp_states[:, k] = dnorm.rvs()

    estden = scipy.stats.gaussian_kde(temp_states)

    kldiv = 0.0
    for k in range(N):
        # draw from samplesw
        kldiv += -dnorm.logpdf(temp_states[:, k]) + estden.logpdf([temp_states[1, k], temp_states[2, k]])

    return (1.0/N)*kldiv


def kluniform(m, S, temp_states, N):

    s11 = S[0, 0]
    s22 = S[1, 1]

    dnorm = scipy.stats.multivariate_normal(mean=m, cov=S)

    m1 = [m[0] - numpy.sqrt(s11)*2, m[0] + numpy.sqrt(s11)*2]
    m2 = [m[1] - numpy.sqrt(s22)*2, m[1] + numpy.sqrt(s22)*2]

    dnorm1 = scipy.stats.uniform()
    dnorm2 = scipy.stats.uniform()

    for k in range(N):
        temp_states[0, k] = dnorm1.rvs() * (max(m1) - min(m1)) + min(m1)
        temp_states[1, k] = dnorm2.rvs() * (max(m2) - min(m2)) + min(m2)

    estden = scipy.stats.gaussian_kde(temp_states)

    kldiv = 0.0
    for k in range(N):
        # draw from samplesw
        kldiv += -dnorm.logpdf(temp_states[:, k]) + estden.logpdf([temp_states[1, k], temp_states[2, k]])

    return (1.0/N)*kldiv


def show_estimated_density(part_states, part_weights, temp_states):
    """Discrete Kullback-Leibler divergence test wrt a multivariate normal model."""

    sweights = 1.0 - sum(part_weights)
    N = len(part_weights)
    if sweights < 0.0:  # add some robustness here (very crude)
        part_weights = part_weights + (1.0/N)*sweights
        print("Particle weights adjusted by ", sweights, " in Auxiliary!")

    elif sweights > 0.0:
        part_weights = part_weights + (1.0/N)*sweights
        print("Particle weights adjusted by ", sweights, " in Auxiliary!")

    for k in range(N):
        j = numpy.random.choice(range(len(part_weights)), size=1, p=part_weights)
        temp_states[:, k] = part_states[:, j]

    estden = scipy.stats.gaussian_kde(temp_states)
    mpc.rc("text", usetex=True)
    mpc.rc("font", family="serif", serif="Computer Modern", size=14)

    plt.figure()  # new figure otherwise very cluttered
    plt.contour(estden)
    plt.xlabel(r"C_A [kmol.m^{-3}]")
    plt.ylabel(r"T_R [K]")


def remove_outliers(xs, multiple=2):
    """remove columns if the column has an element which is more than twice as the mean"""
    rows, cols = xs.shape
    m1 = numpy.average(xs, axis=1)
    remindex = []
    for k in range(cols):
        if xs[0, k] > m1[0]*multiple or xs[1, k] > m1[1]*multiple:
            remindex.append(k)

    fxs = numpy.zeros([rows, cols-len(remindex)])
    counter = 0
    for k in range(cols):
        if not (k in remindex):
            fxs[:, counter] = xs[:, k]
            counter += 1

    return fxs


def model_average(switches, models):
    n = len(switches)
    A = numpy.zeros([2, 2])
    B = numpy.zeros(2)
    b = numpy.zeros(2)
    for k in range(n):
        A += A + models[k].A @ switches[k]
        B += B + models[k].B @ switches[k]
        b += b + models[k].b @ switches[k]

    return A, B, b
