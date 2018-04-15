#  LQR Controller
import numpy
import collections


class Controller:
    def __init__(self, K, x_off, u_off):
        self.K = K
        self.x_off = x_off
        self.u_off = u_off


def lqr(A, B, Q, R):
    """Returns the infinite horizon LQR solution.
    Don't confuse the weighting matrices Q and R with
    the noise covariance matrices!"""
    P = dare(A, B, Q, R)
    F = numpy.linalg.inv(R+B @ P @ B.T) @ B @ P @ A
    return F


def dare(A, B, Q, R):
    """Solves the discrete algebraic ricatti equation (dare)"""
    nr, nc = A.shape
    P = numpy.eye(nr)
    counter = 0
    while True:
        counter += 1
        if isinstance(B @ P @ B.T + R, collections.Iterable):
            inverse = numpy.linalg.inv(B @ P @ B.T + R)
        else:
            inverse = numpy.matrix(1 / (B @ P @ B.T + R))
        Pnow = Q + A.T @ P @ A - A.T @ P @ B.T @ inverse @ B @ P @ A
        if numpy.linalg.norm(Pnow-P, ord=numpy.inf) < 1E-06:
            P = Pnow
            return P
        else:
            P = Pnow

        if counter > 1000000:
            raise ValueError("DARE did not converge...")


def inv(x):
    if len(x) == 1:
        return 1./x[1]
    else:
        raise ValueError("Cannot invert a vector with more than one entry!")


def offset(A, B, C, H, ysp):
    """Returns the state and controller offset."""
    B = numpy.matrix(B)
    B = B.T
    rA, cA = get_size(A)
    rB, cB = B.shape
    rC, cC = get_size(C)
    rH, cH = H.shape
    lenysp, _ = ysp.shape
    if cA+cB-cC == 0:
        z1 = numpy.zeros(rA+rH-rB)
    else:
        z1 = numpy.zeros([rA+rH-rB, cA+cB-cC])

    z2 = numpy.matrix(numpy.zeros(rA+rH-lenysp))
    ssvec = numpy.matrix(numpy.hstack([z2, ysp]))
    ssmat = numpy.vstack([numpy.hstack([numpy.eye(rA)-A, -B]), numpy.hstack([H @ C, z1])])

    ss = numpy.array((numpy.linalg.inv(ssmat) @ ssvec.T).T)[0]
    x_off = ss[:rA]
    u_off = ss[rA:]
    return x_off, u_off


def get_size(A):
    # wraps size() but with added robustness

    if len(A) == 1:
        r = 1
        c = 1
    else:
        x = A.shape
        if len(x) == 1:
            r = x[0]
            c = 0
        else:
            r = x[0]
            c = x[1]

    return r, c
