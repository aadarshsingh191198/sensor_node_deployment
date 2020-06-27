from platypus import NSGAII, Problem, Real, Binary
from random import randint,random 
from math import sqrt
import numpy  as np
import matplotlib.pyplot as plt


#  def init_nodes():
N=40
K=20
m=100

sx = [randint(0,m) for i in range(N)]   #bkue
sy = [randint(0,m) for i in range(N)]
rcom = 12
rsen = 20
tx = [randint(0,m) for i in range(K)]   #black
ty = [randint(0,m) for i in range(K)]
bsx = randint(40,60)    #red
bsy = randint(40,60)
er = [random() for i in range(N)]
# g = [randint(0,1) for i in range(N)]
g = None
# def draw_active(g_int):

def set_g(g_int):
    global g
    if not g:
        print('Nothing in g yet')
    else:
        print(g)
    g= g_int
    print(g)


def distanceSS(i,j):
    return sqrt((sx[i] - sx[j])**2 +(sy[i]-sy[j])**2 )

def distanceST(i,j):
    return sqrt((sx[i]-tx[j])**2 +(sy[i]-ty[j])**2 )

def distanceSB(i):
    return sqrt((sx[i] - bsx)**2 +(sy[i]-bsy)**2 )

def emin(g_int):
    return min([er_ for (er_,g_int_) in zip(er,g_int) if g_int_])

def eta_s(g_int):
    nu_s = []
    for i in range(N):
        nu_si = set()
        for j in range(N):
            if j==i: 
                continue
            
            if g_int[j] and distanceSS(i,j)<= rcom and distanceSB(i) >= distanceSB(j):
                nu_si.add(j)
        nu_s.append(nu_si)

    eta_s = [1 if not len(nu_s_)==0 else -1 for nu_s_ in nu_s]
    return eta_s
def gamma_lam(g_int):
    xi_lam = [] # ya that swiggly e thing is called xi 
    for i in range(K):
        xi_lami = set()
        for j in range(N):
            if i == j:
                continue
            
            if g_int[j] and distanceST(j,i) <= rsen: #Distance of sensor j from target i 
                xi_lami.add(j)
        xi_lam.append(xi_lami)
    
    gamma_lam = [1 if not len(xi_lam_)==0 else -1 for xi_lam_ in xi_lam]
    return gamma_lam


def multi_obj(vars):
    # x = vars[0]
    # y = vars[1]
    w1,w2,w3,w4 = 1,1,1,1
    g_int = list(map(int, vars[0]))
    set_g(g_int)
    o1 = w1*(1-sum(g_int)/N) 
    o2 = w2*sum(gamma_lam(g_int))/K
    o3 = w3*np.sum(np.dot(g_int, eta_s(g_int)))/N
    o4 = w4*emin(g_int)/max(er)
    # draw_active(g_int)
    return [o1+o2+o3+o4]#, [-x + y - 1, x + y - 7]

problem = Problem(1,1) #  decision variables,  objectives, and  constraints, 
problem.types[:] = Binary(N)
problem.directions[:] = Problem.MAXIMIZE
# problem.constraints[:] = "<=0"
problem.function = multi_obj

algorithm = NSGAII(problem)
algorithm.run(100)

print(algorithm.result)

# for solution in algorithm.result:
#     # print(solution)
#     print(solution.objectives)

from circle_plot import Plotter
print(g)
plotter = Plotter(True, N,K,m, g, sx,sy, rcom, rsen, tx,ty, bsx, bsy, er)
plotter.plot_sensors()

plt.scatter( range(len(algorithm.result)), [s.objectives[0] for s in algorithm.result])
# plt.ylim([0, 1.1])
# plt.ylim([0, 1.1])
plt.ylabel("$f_1(x)$")
plt.xlabel("Iterations")
plt.show()