import random, numpy
from . import HMM

class House:
    def __init__(self, n):
        self.n = n
        self.creaks = numpy.random.randint(0, 2, size = (n,n))
        self.bumps = numpy.random.randint(0, 2, size = (n,n))
        self.burglar = int(random.random()*self.n)
        self.burglar = 15
        self.floor = numpy.zeros([n,n])
        self.floor[self.burglar//n][self.burglar%n] = 1

    def createHMM(self):
        tp = numpy.zeros([self.n**2, self.n**2])
        ep = numpy.zeros([4, self.n**2])
        # Transmission probability tables
        for col in range(self.n**2):
            legalMoves = self.getLegalMoves(col)
            prob = 1/len(legalMoves)
            for row in range(self.n**2):
                if row in legalMoves:
                    tp[row][col] += prob

        # Emission probability tables
        cl = 0.9 # creak if light
        ncl = 0.1
        cd = 0.01 # creak if dark
        ncd = 0.99
        bl = 0.9 # bump if light
        nbl = 0.1
        bd = 0.01 # bump if dark
        nbd = 0.99
        # Emission 1: creak and bump
        # Emission 2: creak but no bump
        # Emission 3: no creak but a bump
        # Emission 4: neither a creak nor a bump
        for col in range(self.n**2):
            r, c = col//self.n, col%self.n 
            if self.creaks[r][c] == 1 and self.bumps[r][c] == 1: # light light
                ep[0][col], ep[1][col], ep[2][col], ep[3][col] = [cl*bl, cl*nbl, ncl*bl, ncl*nbl]
            elif self.creaks[r][c] == 1 and self.bumps[r][c] == 0: # light dark
                ep[0][col], ep[1][col], ep[2][col], ep[3][col] = [cl*bd, cl*nbd, ncl*bd, ncl*nbd]
            elif self.creaks[r][c] == 0 and self.bumps[r][c] == 1: # dark light
                ep[0][col], ep[1][col], ep[2][col], ep[3][col] = [cd*bl, cd*nbl, ncd*bl, ncd*nbl]
            else: # dark dark
                ep[0][col], ep[1][col], ep[2][col], ep[3][col] = [cd*bd, cd*nbd, ncd*bd, ncd*nbd]

        return HMM.HMM(tp, ep)

    def getLegalMoves(self, k):
        # Returns an array of legal moves
        moves = []
        loc = k

        up = loc-1
        down = loc+1
        left = loc - self.n
        right = loc + self.n

        up in range(self.n**2) and not(loc in range(0,self.n**2,self.n)) and moves.append(up)
        down in range(self.n**2) and not (loc in range(self.n-1,self.n**2, self.n)) and moves.append(down)
        left in range(self.n**2) and moves.append(left)
        right in range(self.n**2) and moves.append(right)
        return moves

    def move(self):
        # Move the burglar
        moves = self.getLegalMoves(self.burglar)
        self.floor[self.burglar//self.n][self.burglar%self.n] = 0
        self.burglar = moves[int(random.random()*len(moves))]
        self.floor[self.burglar//self.n][self.burglar%self.n] = 1

    def getLocation(self):
        return self.burglar
