# switching particle filter
import numpy
import copy
import scipy.stats
import typing
import collections


class Particles:
    def __init__(self, x, s, w):
        self.x = x  # states
        self.s = s  # switches
        self.w = w  # weights
        

class Model:
    def __init__(self, F, G, A, xdists, ydists):
        self.F = F  # transition
        self.G = G  # emission
        self.A = A  # HMM model, columns sum to 1
        self.xdists = xdists
        self.ydists = ydists


def init_spf(xdist, sdist, nP, xN):
    """Initialise the particle filter.
    xdist => state prior (a distribution from the package Distributions)
    sdist => switch prior distribution
    nP => number of particles
    nX => number of states per particle
    Return an array of nP particles"""

    particles = Particles(numpy.zeros([xN, nP]), numpy.zeros(nP, dtype=numpy.int64), numpy.zeros(nP))
    for p in range(nP):
        xdraw = xdist.rvs()
        sdraw = numpy.random.choice(range(len(sdist)), p=sdist)
        particles.x[:, p] = xdraw
        particles.s[p] = sdraw
        particles.w[p] = 1/nP  # uniform initial weight

    return particles


def init_filter(particles, y, model):

    nX, N = particles.x.shape
    nS, _ = model.A.shape

    for p in range(N):
        for s in range(nS):
            if particles.s[p] == s:
                if not isinstance(y, collections.Iterable):
                    particles.w[p] = particles.w[p]*model.ydists[s].pdf(y - model.G[s](particles.x[:, p]))
                else:
                    temp = numpy.subtract(y, model.G[s](particles.x[:, p]))
                    particles.w[p] = particles.w[p] * model.ydists[s].pdf(temp)

    # particles.w = particles.w .+ abs(minimum(particles.w)) #no negative number issue
    particles.w /= sum(particles.w)

    if number_effective_particles(particles) < N/2:
        particles = resample(particles)
    return particles


def spf_filter(particles, u, y, model):
    nX, N = particles.x.shape
    nS, _ = model.A.shape

    # This can be made more compact but at the cost of clarity
    # first draw switch sample
    for p in range(N):
        for s in range(nS):
            if particles.s[p] == s:
                particles.s[p] = numpy.random.choice(range(len(model.A[:, s])), size=1, p=model.A[:, s])
                # rand(Categorical(model.A[:,s]))
    # Now draw (predict) state sample
    for p in range(N):
        for s in range(nS):
            if particles.s[p] == s:
                noise = model.xdists[s].rvs()
                particles.x[:, p] = model.F[s](particles.x[:, p], u, noise)  # predict
                particles.w[p] = particles.w[p]*model.ydists[s].pdf(y - model.G[s](particles.x[:, p]))

                if numpy.isnan(particles.w[p]):
                    print("Particle weight issue...")
                    particles.w[p] = 0

    # particles.w = particles.w .+ abs(minimum(particles.w)) #no negative number issue
    if max(particles.w) < 1/(N**2):
        print("The particles all have very small weight...")
    particles.w /= sum(particles.w)
    for w in particles.w:
        if numpy.isnan(w):
            raise ValueError("Particles have become degenerate!")

    if number_effective_particles(particles) < N/2:
        particles = resample(particles)
        
    return particles
    
    
def resample(particles):
    N = len(particles.w)
    # sample = rand(Categorical(particles.w), N)  # draw N samples from weighted Categorical
    sample = numpy.random.choice(range(N), size=N, p=particles.w)
    copyparticles_x = copy.copy(particles.x)
    copyparticles_s = copy.copy(particles.s)
    for p in range(N):  # resample
        particles.x[:, p] = copyparticles_x[:, sample[p]]
        particles.s[p] = copyparticles_s[sample[p]]
        particles.w[p] = 1/N
    particles = roughen(particles)
    return particles


def number_effective_particles(particles):
    """Return the effective number of particles."""
    N = len(particles.w)
    numeff = 0
    for p in range(N):
        numeff += particles.w[p]**2
    
    return 1/numeff


def roughen(particles):
    """Roughening the samples to promote diversity"""
    xN, N = particles.x.shape
    sig = numpy.zeros(xN)

    K = 0.2  # parameter...

    for k in range(xN):
        D = max(particles.x[k, :]) - min(particles.x[k, :])
        if D == 0:
            print("Particle distance very small! Roughening could cause problems...")
        sig[k] = K*D*N**(-1/xN)

    sigma = numpy.diag(sig**2)
    jitter = scipy.stats.multivariate_normal(cov=sigma)
    for p in range(N):
        particles.x[:, p] = particles.x[:, p] + jitter.rvs()
    
    return particles


def get_stats(particles):
    """Return the Gaussian statistics of the particles."""
    mean = numpy.average(particles.x, weights=particles.w, axis=1)
    cov = numpy.cov(particles.x, aweights=particles.w)
    return mean, cov
    

def calc_a(linsystems):
    """Returns a stochastic HMM matrix based on [need a good way to do this!!!]"""
    N = len(linsystems)
    A = numpy.zeros([N, N])  # pre-allocate A

    for j in range(N):
        for i in range(N):
            A[i, j] = numpy.linalg.norm((linsystems[i].op-linsystems[j].op) / linsystems[j].op)
            # A[i,j] = norm(linsystems[i].op-linsystems[j].op)

    posA = numpy.zeros([N, N])

    for j in range(N):
        a = numpy.sort(A[:, j])[::-1]
        for i in range(N):
            posA[i, j] = numpy.where(a == A[i, j])[0][0]

    posA /= sum(posA, 1)

    return posA


def get_f(linsystems):
    """Return transmission function matrices"""
    N = len(linsystems)
    F = [None]*N  # type: typing.List[(typing.Any, typing.Any)->typing.Any]
    for k in range(N):
        def f(x, u, w):
            return linsystems[k].A @ x + linsystems[k].B*u + w

        F[k] = f
    return F


def get_g(linsystems, C):
    """Return emission function matrices"""
    N = len(linsystems)
    G = [None]*N  # type: typing.List[typing.Any->typing.Any]
    for k in range(N):
        def g(x):
            return C @ x
        G[k] = g

    return G


def get_dists(linsystems, dist):
    N = len(linsystems)
    xdists = [dist]*N

    return xdists


def get_max_track(particles, numSwitches):
    maxtrack = numpy.zeros(numSwitches)
    numParticles = len(particles.w)
    totals = numpy.zeros(numSwitches)

    for p in range(numParticles):
        totals[particles.s[p]] += particles.w[p]

    maxtrack[numpy.argmax(totals)] = 1.0
    return maxtrack

