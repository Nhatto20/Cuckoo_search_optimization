import numpy as np
import matplotlib.pyplot as plt
from math import gamma



class CSO:

    def __init__(self, fitness,n=2,  pa=0.25, P=150, beta=1.5, bound=None, min=True, max = False,graph = False, Tmax=100, k = 0.01, printing = False):
        
        self.fitness = fitness
        self.P = P 
        self.n = n
        self.Tmax = Tmax
        self.pa = pa
        self.beta = beta
        self.bound = bound
        self.graph = graph
        self.min = min
        self.max = max
        
        self.k = k
        self.printing = printing
        
        self.X = []
        
        if bound is not None:
            for (U, L) in bound:
                x = (U-L)*np.random.rand(P,) + L 
                self.X.append(x)
            
            self.X = np.array(self.X).T
        else:
            self.X = np.random.randn(P,n)

    def update_position_1(self):

        num = gamma(1+self.beta)*np.sin(np.pi*self.beta/2)
        den = gamma((1+self.beta)/2)*self.beta*(2**((self.beta-1)/2))
        ou = (num/den)**(1/self.beta)
        ov = 1
        u = np.random.normal(0, ou, self.n)
        v = np.random.normal(0, ov, self.n)
        S = u/(np.abs(v)**(1/self.beta))


        for i in range(self.P):
            if i==0:
                self.best = self.X[i,:].copy()
            else:
                self.best = self.optimum(self.best, self.X[i,:])

        newX = self.X.copy()
        for i in range(self.P):
            newX[i,:] += np.random.randn(self.n)*self.k*S*(newX[i,:]-self.best) 
            self.X[i,:] = self.optimum(newX[i,:], self.X[i,:])

    def update_position_2(self):

        newX = self.X.copy()
        oldX = self.X.copy()
        for i in range(self.P):
            d1,d2 = np.random.randint(0,5,2)
            for j in range(self.n):
                r = np.random.rand()
                if r < self.pa:
                    newX[i,j] += np.random.rand()*(oldX[d1,j]-oldX[d2,j]) 
            
            self.X[i,:] = self.optimum(newX[i,:], self.X[i,:])
    

    def optimum(self, best, particle_x):
        
        if self.min==True and self.max ==False:
            if self.fitness(best) > self.fitness(particle_x):
                best = particle_x.copy()
        else:
            if self.fitness(best) < self.fitness(particle_x):
                best = particle_x.copy()
        return best


    def clip_X(self):

        if self.bound is not None:
            for i in range(self.n):
                xmin, xmax = self.bound[i]
                self.X[:,i] = np.clip(self.X[:,i], xmin, xmax)

    def execute(self):

        self.fitness_time, self.time = [], []

     
        for t in range(self.Tmax):
            self.update_position_1()
            self.clip_X()
            self.update_position_2()
            self.clip_X()
            self.fitness_time.append(self.fitness(self.best))

            self.time.append(t)
            if self.printing:
                print(t+1,'| best global fitness:',round(self.fitness(self.best),7))
            
        if self.printing:
            print('\nOptimal Solution: ', np.round(self.best.reshape(-1),7))
            print('\nOptimal Fitness:', np.round(self.fitness(self.best),7))
        if self.graph:
            self.make_graph()

        return self.best, self.fitness(self.best)
        
    def make_graph(self):
        
        plt.plot(self.time, self.fitness_time)
        plt.title('Fitness value vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness value')
        plt.show()

def bmi(X):
    a = X[0]
    b = X[1]
    x = np.array([1.67 , 1.74 , 1.70, 1.76])
    y = np.array([55,93,70,76])
    y_pred = a*x + b
    return np.sum((y_pred - y)**2 ) *0.5


#m,n = CSO(fitness= bmi, printing = True, P = 100, Tmax = 10000).execute()
#print(m)
#print(n)
    