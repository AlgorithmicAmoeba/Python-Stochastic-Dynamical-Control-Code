import numpy, random, scipy.optimize

class LinearReactor:
    def __init__(self, op, A, B, b):
        self.op = op
        self.A = A
        self.B = B
        self.b = b

class Reactor:
    def __init__(self, V=5, R=8.314, CA0=1, TA0=310, dH=-4.78e4, k0=72e7, E=8.314e4, Cp=0.239, rho=1000, F=0.1):
        self.V = V
        self.R = R
        self.CA0 = CA0
        self.TA0 = TA0
        self.dH = dH
        self.k0 = k0
        self.E = E
        self.Cp = Cp
        self.rho = rho
        self.F = F
        
    def run_reactor(self, xprev, u, h):
        # Use Runga-Kutta method to solve for the next time step using the full
        k1 = self.reactor_ode(xprev, u)
        k2 = self.reactor_ode(xprev + 0.5*h*k1, u)
        k3 = self.reactor_ode(xprev + 0.5*h*k2, u)
        k4 = self.reactor_ode(xprev + h*k3, u)
        xnow = xprev + (h/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
        return xnow

    def reactor_ode(self, xprev, u):
        # Evaluate the ODE defs describing the reactor.
        xnow = numpy.zeros(2)
        xnow[0] = (self.F/self.V) * (self.CA0 - xprev[0]) - self.k0*numpy.exp(-self.E/(self.R*xprev[1]))*xprev[0]
        xnow[1] = (self.F/self.V) * (self.TA0 - xprev[1]) 
        xnow[1]-= (self.dH/(self.rho*self.Cp))*self.k0*numpy.exp(-self.E/(self.R*xprev[1]))*xprev[0] 
        xnow[1]+= u/(self.rho*self.Cp*self.V)
        return xnow
        
    def reactor_func(self, xprev, u):
        # Evaluate the ODE defs describing the reactor. In the format required by NLsolve!
        # xnow :: Array{Float64, 1} = zeros(2)
        xnow = [0]*2
        xnow[0] = (self.F/self.V) * (self.CA0 - xprev[0]) - self.k0*numpy.exp(-self.E/(self.R*xprev[1]))*xprev[0]
        xnow[1] = (self.F/self.V) * (self.TA0 - xprev[1])
        xnow[1]-= (self.dH/(self.rho*self.Cp))*self.k0*numpy.exp(-self.E/(self.R*xprev[1]))*xprev[0] 
        xnow[1]+= u/(self.rho*self.Cp*self.V)
        return xnow

    def jacobian(self, x):
        # Returns the Jacobian evaluated at x
        J11 = -self.F/self.V-self.k0*numpy.exp(-self.E/(self.R*x[1]))
        J12 = -x[0]*self.k0*numpy.exp(-self.E/(self.R*x[1]))*(self.E/(self.R*x[1]**2))
        J21 = -self.dH/(self.rho*self.Cp)*self.k0*numpy.exp(-self.E/(self.R*x[1]))
        J22 = -(self.F/self.V + self.dH/(self.rho*self.Cp)*self.k0*numpy.exp(-self.E/(self.R*x[1]))*(self.E/(self.R*x[1]**2))*x[0])
        return numpy.array([[J11, J12], [J21, J22]])

    def QG(self, T):
        # Return the evaluated heat generation term.
        ca = self.F/self.V*self.CA0/(self.F/self.V + self.k0*numpy.exp(-self.E/(self.R*T)))
        qg = -self.dH/(self.rho*self.Cp)*self.k0*numpy.exp(-self.E/(self.R*T))*ca
        return qg

    def CA(self, T):
        ca = self.F/self.V*self.CA0/(self.F/self.V + self.k0*numpy.exp(-self.E/(self.R*T)))
        return ca

    def QR(self, T, Q):
        # Return the evaluated heat removal term.
        qr = - self.F/self.V*(self.TA0 - T) - Q/(self.rho * self.V * self.Cp)
        return qr
        
    def linearise(self, linpoint, h):
        # Returns the linearised coefficients of the Runge Kutta method
        # given a linearisation point.
        # To solve use x(k+1) =  Ax(k) + Bu(k)
        B11 = 0.0
        B21 = 1.0/(self.rho*self.V*self.Cp)
        B = [B11, B21]
        A = self.jacobian(linpoint)
        F0 = self.reactor_ode(linpoint, 0.0) # u = 0 because Bs account for the control term
        D = numpy.matmul(A, linpoint)
        # now we have x' = Ax + Bu + F0 - D = F(x)
        # now write ito deviation variables!
        newb = numpy.matmul(numpy.linalg.inv(A), D-F0)
        # now we have xp' = Axp + Bu where x = xp + newb

        n = len(A)
        # Uses the bilinear transform aka the Tustin transform... google it...
        newA = numpy.matmul((numpy.identity(n) + 0.5*h*A), numpy.linalg.inv(numpy.identity(n) - 0.5*h*A))
        newB = numpy.matmul(numpy.matmul(numpy.linalg.inv(A), (newA-numpy.identity(n))), B)

        return newA, newB, newb

    def discretise(self, nX, nY, xspace, yspace):
        # Discrete the state space into nX*nY regions.
        dx = (xspace[2] - xspace[1])/nX
        dy = (yspace[2] - yspace[1])/nY

        operatingpoints = numpy.zeros([2,nX*nY + 3]) # add the three nominal points

        k = 1 #counter
        for x in range(nX):
            xnow = dx*(x-1) + xspace[1] + dx*0.5
            for y in range(nY):
                ynow = dy*(y-1) + yspace[1] + dy*0.5
                operatingpoints[:, k] = [xnow, ynow]
                k += 1
        operatingpoints[:, k] = [0.009718824131074055, 508.0562351737852]
        operatingpoints[:, k+1] = [0.48934869384879404, 412.1302612302412]
        operatingpoints[:, k+2] = [0.9996453064079288, 310.07093871841454]
        return operatingpoints

    def discretise_randomly(self, npoints, xspace, yspace):
        # Perform the same action as discretise() except pick points to
        # discretise around at random.
        operatingpoints = numpy.zeros([2, npoints+3])
        if npoints==0:
            k = 0
        else:
            k=1
        for k in range(npoints):
            nx = random.rand()
            ny = random.rand()
            xnow = xspace[1] + nx*(xspace[2] - xspace[1])
            ynow = yspace[1] + ny*(yspace[2] - yspace[1])
            operatingpoints[:, k] = [xnow, ynow]

        operatingpoints[:, k] = [0.009718824131074055, 508.0562351737852]
        operatingpoints[:, k+1] = [0.48934869384879404, 412.1302612302412]
        operatingpoints[:, k+2] = [0.9996453064079288, 310.07093871841454]
        return operatingpoints

    def getLinearSystems(self, nX, nY, xspace, yspace, h):
        # Returns an array of linearised systems

        N = nX*nY + 3 # add the three nominal operating points
        linsystems = [None]*N
        ops = discretise(nX, nY, xspace, yspace)
        for k in range(N):
            op = ops[:, k]
            A, B, b = linearise(op, h)
            linsystems[k] = LinearReactor(op, A, B, b)
            
        return linsystems

    def getLinearSystems_randomly(self, npoints, xspace, yspace, h):
        # Returns an array of linearised systems

        N = npoints + 3 # add the three nominal operating points
        linsystems = [None]*N
        ops = self.discretise_randomly(npoints, xspace, yspace)
        for k in range(N):
            op = ops[:, k]
            A, B, b = self.linearise(op, h)
            linsystems[k] = LinearReactor(op, A, B, b)

        return linsystems

    def getNominalLinearSystems(self, h):
        # Returns an array of linearised systems

        linsystems = [None]*3

        #Get the steady state points
        xguess1 = [0.073, 493.0]
        xguess2 = [0.21, 467.0]
        xguess3 = [0.999, 310.0]
        f = lambda x: self.reactor_func(x, 0.0)

        xx1res = scipy.optimize.fsolve(f, xguess1)
        xx2res = scipy.optimize.fsolve(f, xguess2)
        xx3res = scipy.optimize.fsolve(f, xguess3)

        ops = numpy.zeros([len(xx1res), 3])
        ops[:, 0] = xx1res
        ops[:, 1] = xx2res
        ops[:, 2] = xx3res

        for k in range(3):
            op = ops[:, k]
            A, B, b = self.linearise(op, h)
            linsystems[k] = LinearReactor(op, A, B, b)

        return linsystems
