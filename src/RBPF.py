# Rao Blackwellised Particle Filter
# WARNING: this is made specifically for the system I am investigating
import numpy
import scipy.stats
import copy
import src.SPF as SPF

print("RBPF is hardcoded for the CSTR!")


class Particles:
    def __init__(self, mus, sigmas, ss, ws):
        self.mus = mus  # mean
        self.sigmas = sigmas  # covarience
        self.ss = ss  # switches
        self.ws = ws  # weights


class Model:
    def __init__(self, A, B, b, C, Q, R):
        self.A = A
        self.B = B
        self.b = b
        self.C = C
        self.Q = Q
        self.R = R


def setup_rbpf(linsystems, C, Q, R):
    """Setup each switch"""
    N = len(linsystems)
    models = numpy.array([None]*N)
    for k in range(N):
        models[k] = Model(linsystems[k].A, linsystems[k].B, linsystems[k].b, C, Q, R)
        
    A = SPF.calc_a(linsystems)
    return models, A
    

def init_rbpf(sdist, mu_init, sigma_init, xN, nP):
    """Initialise the particle filter."""

    particles = Particles(numpy.zeros([xN, nP]), numpy.zeros([xN, xN, nP]), numpy.zeros(nP), numpy.zeros(nP))
    for p in range(nP):
        sdraw = numpy.random.choice(range(len(sdist)), p=sdist)
        particles.mus[:, p] = mu_init  # normal mu
        particles.sigmas[:, :, p] = sigma_init
        particles.ss[p] = sdraw
        particles.ws[p] = 1/nP  # uniform initial weight

    return particles


def init_filter(particles, u, y, models):

    nX, N = particles.mus.shape
    nS = len(models)

    for p in range(N):
        for s in range(nS):
            if particles.ss[p] == s:
                particles.mus[:, p] = particles.mus[:, p] - models[particles.ss[p]].b  # adjust mu for specific switch
                mu = models[s].C @ (models[s].A @ particles.mus[:, p] + models[s].B*u)
                temp = (models[s].A @ particles.sigmas[:, :, p] @ models[s].A.T + models[s].Q)
                sigma = models[s].C @ temp @ models[s].C.T + models[s].R

                d = scipy.stats.multivariate_normal(mean=mu, cov=sigma)
                if len(y) == 1:
                    particles.ws[p] = particles.ws[p]*d.pdf([y-models[s].b[2]])  # HARDCODED for this system!!!
                else:
                    particles.ws[p] = particles.ws[p]*d.pdf(y-models[s].b)  # weight of each particle
            # println("Switch: ", s, " Predicts: ", round(mu + models[s].b, 4), "Observed: ", round(y,4),
            # " Weight: ", round(particles.ws[p], 5))
            particles.mus[:, p] = particles.mus[:, p] + models[particles.ss[p]].b  # fix mu for specific switch

    particles.ws /= sum(particles.ws)

    if number_effective_particles(particles) < N/2:
        particles = resample(particles)

    return particles


def rbpf_filter(particles, u, y, models, A):

    nX, N = particles.mus.shape
    nS = len(models)

    # This can be made more compact but at the cost of clarity
    # first draw switch sample
    for p in range(N):
        for s in range(nS):
            if particles.ss[p] == s:
                particles.ss[p] = numpy.random.choice(range(len(A[:, s])), size=1, p=A[:, s])


# apply KF and weight
    for p in range(N):
        for s in range(nS):
            if particles.ss[p] == s:
                mu = models[s].C @ (models[s].A @ (particles.mus[:, p] - models[s].b) + models[s].B*u)
                temp = (models[s].A @ particles.sigmas[:, :, p] @models[s].A.T + models[s].Q)
                sigma = models[s].C @ temp @ models[s].C.T + models[s].R
                d = scipy.stats.multivariate_normal(mean=mu, cov=sigma)
                if len(y) == 1:
                    particles.ws[p] = particles.ws[p]*d.pdf([y-models[s].b[2]])  # HARDCODED for this system!!!
                else:
                    particles.ws[p] = particles.ws[p]*d.pdf(y-models[s].b)  # weight of each particle

                if numpy.isnan(particles.ws[p]):
                    print("Particle weight issue...")
                    particles.ws[p] = 0.0

                # println("Switch: ", s, " Predicts: ", round(mu + models[s].b, 4), "Observed: ", round(y,4),
                # " Weight: ", round(particles.ws[p], 5))

                pmean = models[s].A @ (particles.mus[:, p] - models[s].b) + models[s].B*u
                pvar = models[s].Q + models[s].A @ particles.sigmas[:, :, p] @ models[s].A.T
                kalmanGain = pvar @ models[s].C.T @ numpy.linalg.inv(models[s].C @ pvar @ models[s].C.T + models[s].R)
                ypred = models[s].C @ pmean  # predicted measurement
                if len(y) == 1:
                    updatedMean = pmean + kalmanGain @ (y - models[s].b[1] - ypred)  # adjust for state space
                else:
                    updatedMean = pmean + kalmanGain @ (y - models[s].b - ypred)  # adjust for state space

                rows, cols = pvar.shape
                updatedVar = (numpy.eye(rows) - kalmanGain @ models[s].C) @ pvar

                particles.sigmas[:, :, p] = updatedVar
                particles.mus[:, p] = updatedMean + models[s].b  # fix

    # particles.ws = particles.ws .+ abs(minimum(particles.ws)) #no negative number issue
    if max(particles.ws) < (1.0/(N**2)):
        print("The particles all have very small weight...")

    if sum(particles.ws) == 0.0:
        raise ValueError("Zero cumulative weight...")

    particles.ws /= sum(particles.ws)

    for w in particles.ws:
        if numpy.isnan(w):
            raise ValueError("Particles have become degenerate! (after normalisation)")

    if number_effective_particles(particles) < N/2:
        particles = resample(particles)

    return particles


def resample(particles):
    N = len(particles.ws)
    sample = numpy.random.choice(range(len(particles.ws)), size=N, p=particles.ws)
    copyparticles_mus = copy.copy(particles.mus)
    copyparticles_sigmas = copy.copy(particles.sigmas)
    copyparticles_ss = copy.copy(particles.ss)
    for p in range(N):  # resample
        particles.mus[:, p] = copyparticles_mus[:, sample[p]]
        particles.sigmas[:, :, p] = copyparticles_sigmas[:, :, sample[p]]
        particles.ss[p] = copyparticles_ss[sample[p]]
        particles.ws[p] = 1/N

    particles = roughen(particles)
    return particles


def number_effective_particles(particles):
    """Return the effective number of particles."""
    N = len(particles.ws)
    numeff = 0.0
    for p in range(N):
        numeff += particles.ws[p]**2

    return 1/numeff


def roughen(particles):
    """Roughening the samples to promote diversity"""
    xN, N = particles.x.shape
    sig = numpy.zeros(xN)

    K = 0.2  # parameter...
    flag = True
    for k in range(xN):
        D = max(particles.mus[k, :]) - min(particles.mus[k, :])
        if D == 0.0:
            print("Particle distance very small! Roughening could cause problems...")
            flag = False
            break
        sig[k] = K*D*N**(-1./xN)

    if flag:
        sigma = numpy.diag(sig**2)
        jitter = scipy.stats.multivariate_normal(cov=sigma)
        for p in range(N):
            particles.mus[:, p] = particles.mus[:, p] + jitter.rvs()

    return particles


def get_ave_stats(particles):
    nX, nP = particles.mus.shape
    ave = numpy.zeros(nX)
    avesigma = numpy.zeros([nX, nX])
    for p in range(nP):
        ave = ave + particles.ws[p]*particles.mus[:, p]
        avesigma = avesigma + particles.ws[p]  @ particles.sigmas[:, :, p]

    return ave, avesigma


def get_ml_stats(particles):
    nX, nP = particles.mus.shape
    mlmu = numpy.zeros(nX)
    mlsigma = numpy.zeros([nX, nX])
    prevmaxweight = 0.0
    for p in range(nP):
        if particles.ws[p] > prevmaxweight:
            mlmu = particles.mus[:, p]
            mlsigma = particles.sigmas[:, :, p]
            prevmaxweight = particles.ws[p]

    return mlmu, mlsigma


def get_max_track(particles, numSwitches):
    maxtrack = numpy.zeros(numSwitches)
    numParticles = len(particles.ws)
    totals = numpy.zeros(numSwitches)
    for p in range(numParticles):
        totals[particles.ss[p]] += particles.ws[p]

    maxtrack[numpy.argmax(totals)[0]] = 1.0
    return maxtrack


def smoothed_track(numSwitches, switchtrack, ind, N):
    """returns a smoothed version of maxtrack given the history of the switch movements"""
    sN = get_history(ind, N)  # number of time intervals backwards we look
    modswitchtrack = sum(switchtrack[:, ind-sN:ind], 2)
    modmaxtrack = numpy.zeros(numSwitches)
    modmaxtrack[numpy.argmax(modswitchtrack)[0]] = 1.0
    return modmaxtrack


def get_history(ind, N):
    history = 0
    flag = True

    for k in range(N):
        if ind-k == 0:
            history = k
            flag = False
            break

    if flag:
        history = N

    return history-1


def get_initial_switches(initial_states, linsystems):
    N = len(linsystems)
    initstates = numpy.zeros(N)  # pre-allocate

    for i in range(N):
        initstates[i] = numpy.linalg.norm(linsystems[i].op-initial_states)

    a = numpy.sort(initstates)[:: -1]
    posA = numpy.zeros(N)
    for i in range(N):
        posA[i] = numpy.where(a == initstates[i])[0][0]

    posA /= sum(posA, 1)

    return posA
