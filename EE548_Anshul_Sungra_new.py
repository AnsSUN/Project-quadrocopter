# -*- coding: utf-8 -*-
"""
Created on Thu Mar 08 14:50:45 2018

@author: anshu
"""

import numpy as np
import matplotlib as plt
import control as ctrl
import _547 as lsg
from scipy import linalg as la
import scipy as sci
import numpy.linalg as sla
np.set_printoptions(precision=2)

g,m,I = 9.81,1.,1. # m/sec^2, kg, kg m^2

def f(t,x,u):
    q,dotq = x[:3],x[3:] # positions, velocities
    h,v,theta = q # horiz., vert., rotation
    u1,u2 = u # thrust, torque
    return np.hstack([dotq,(u1/m)*np.sin(theta),
                        -g + (u1/m)*np.cos(theta),
                        u2/I])

def h(t,x,u):
    q,dotq = x[:3],x[3:] # positions, velocities
    h,v,theta = q # horiz., vert., rotation
    return np.array([h,v]) # horizontal, vertical position

q0 = np.array([0.,1.,0.])
dotq0 = np.array([0.,0.,0.])
x0 = np.hstack((q0,dotq0))
u0 = np.array([m*g,0.])
print 'x0 =',x0,'\nf(x0) =',f(0.,x0,u0)

dt = 0.002 # timestep
omg = .5 # one cycle every two seconds
t = 2./omg # two periods
q = [0.,.1,0.] # start 10cm up off the ground
dotq = [0.,0.,0.] # start with zero velocity
x = np.hstack((q,dotq))

# input is a periodic function of time
ut = lambda t : np.array([m*g + np.sin(2*np.pi*t*omg),0.])
# lambda is a shorthand way to define a function
# -- equivalently:
def u(t):
    return np.array([m*g + np.sin(2*np.pi*t*omg),0.])


t_s , x_s = lsg.forward_euler(f,t,x,dt=dt,ut=ut)
u_s = np.array([u(t) for t in t_s])
# x_[j] is the state of the system (i.e. pos. and vel.) at time t_[j]

fig = plt.pyplot.figure();

ax = plt.pyplot.subplot(311)
ax.plot(t_s,x_s[:,:3],'.-')
ax.set_xlabel('time (sec)')
ax.set_ylabel('positions')
ax.legend([r'$h$',r'$v$',r'$\theta$'],ncol=3,loc=2)

ax = plt.pyplot.subplot(312)
ax.plot(t_s,x_s[:,3:],'.-')
ax.set_xlabel('time (sec)')
ax.set_ylabel('velocities')

ax = plt.pyplot.subplot(313)
ax.plot(t_s,u_s,'.-')
ax.set_xlabel('time (sec)')
ax.set_ylabel('inputs')

"""
import _anim as an

fig, ax = plt.pyplot.subplots(figsize=(4,4)); ax.axis('equal'); ax.grid('on');

line, = ax.plot([], [], 'b', lw=2);


def funint():
    gh,gv = [-10.,10.,10.,-10.],[0.,0.,-5.,-.5]
    ax.fill(gh,gv,'blue')
    line.set_data([], [])
    ax.set_xlim(( -1., 1.))
    ax.set_ylim(( -.15, 2.))
    return (line,)


def animate(t):
    i = (t_ >= t).nonzero()[0][0]
    h,v,th = x_[i,:3]
    w = .25
    x = np.array([-w/2.,w/2.,np.nan,0.,0.])
    y = np.array([0.,0.,np.nan,0.,+w/3.])
    z = (x + 1.i*y)*np.exp(1.i*th) + (h + 1.i*v)
    line.set_data(z.real, z.imag)
    return (line,)

plt.pyplot.close(fig)

# call the animator
anim= an.animation.FuncAnimation(fig, animate, init_func=funint, repeat=True,
                        frames=np.arange(0.,t_[-1],.1), interval=20, blit=True)

anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.pyplot.show()
"""

"""
Problem 2
"""

A = lsg.D(lambda x : f(0.,x,u(0)),x)
B = lsg.D(lambda u : f(0.,x,u),u(0))

print 'A =\n',A,'\n','B =\n',B

W = ctrl.lyap(-1.5*np.identity(6)-A,np.dot(B,B.T)) # lyapunov for stability
K = 0.5*np.dot(B.T,la.inv(W)) # for stablize

print "System is Controllable?", lsg.controllable(A,B)

print "closed loop stable?",np.all(np.array(la.eigvals(A - np.dot(B,K))).real
                                   < 0)

ux = lambda x : np.dot(x - x0, -K.T) + u0


np.random.seed(500)

dt = 0.002
t = 3. 
x = x0 + 5*(np.random.rand(6)-0.5)

# input is now a function of state
ux = lambda x : np.dot(x - x0, -K.T) + u0

t_s,x_s = lsg.forward_euler(f,t,x,dt=dt,ux=ux)
u_s = np.array([ux(x) for x in x_s])

fig = plt.pyplot.figure();

ax = plt.pyplot.subplot(311)
ax.plot(t_s,x_s[:,:3],'.-')
ax.set_xlabel('time (sec)')
ax.set_ylabel('positions')
ax.legend([r'$h$',r'$v$',r'$\theta$'],ncol=3,loc=1)

ax = plt.pyplot.subplot(312)
ax.plot(t_s,x_s[:,3:],'.-')
ax.set_xlabel('time (sec)')
ax.set_ylabel('velocities')

ax = plt.pyplot.subplot(313)
ax.plot(t_s,u_s,'.-')
ax.set_xlabel('time (sec)')
ax.set_ylabel('inputs')

"""
Problem 3:
"""
# Defining LQR function:

def CTLQR(A,B,Q,R,Pt,tf,dt=dt):
    t_n = 0.0
    K = []
    P = []
    while t_n < tf:
        P_t = la.solve_continuous_are(A(t_n),B(t_n),Q(t_n),R(t_n))
        K_t = np.dot(sla.inv(R(t_n)), np.dot(B(t_n).T,P_t))
        K.append(K_t)
        P.append(P_t)
        t_n += dt
    K_t = np.dot(sla.inv(R(t_n)), np.dot(B(t_n).T,Pt))
    K.append(K_t)
    P.append(Pt)
    return np.asarray(K),np.asarray(P)

# Defining linear system
def funLin(t,x,u):
    return np.dot(A,x) + np.dot(B,u)

tfin=5.0
dt=0.001

#Defining cost
def Cost(t,x,u):
    J = 0.0
    i = 0
    t_n = 0
    while t_n < t:
        J += (.5*np.dot(np.dot(x[i,:].T,Q(t_n)),x[i,:]) + \
              .5*np.dot(np.dot(u[i,:].T,R(t_n)),u[i,:]))*dt
        t_n += dt
        i += 1
    return J


# Parameters for CTLQR function
Ain = lambda t : A
Bin = lambda t : B
Qg= sci.random.rand(6,6)
Q = lambda t : 100.0*np.dot(Qg,Qg.transpose())
Rg= sci.random.rand(2,2)
R = lambda t : 1.0*np.dot(Rg,Rg.transpose())
Pg= sci.random.rand(6,6)
Pt = 0.0*np.dot(Pg,Pg.transpose())

K,P = CTLQR(Ain,Bin,Q,R,Pt,tfin,dt=dt)
def uOut(t,x):
    return -np.dot(K[int(t/dt)],x)

t_LQR,x_LQR,u_LQR = lsg.forward_euler(f=funLin,t=tfin,
                                      x=np.asarray([1,1.0,0.0,0.0,0.0,0.0]),
                                      utx=uOut,dt=dt,return_u=True)

fig = plt.pyplot.figure();

plt.pyplot.plot(x_LQR[:,0])
plt.pyplot.plot(x_LQR[:,1])
plt.pyplot.plot(x_LQR[:,2])
plt.pyplot.plot(x_LQR[:,3])
plt.pyplot.plot(x_LQR[:,4])
plt.pyplot.plot(x_LQR[:,5])
plt.pyplot.title('LQR Control of quadrotor model')
plt.pyplot.xlabel('Time Steps')
plt.pyplot.ylabel('6 States')
plt.pyplot.legend([r'$h$',r'$v$',r'$\theta$', r'$\dot h$',
                   r'$\dot v$', r'$\dot{\theta}$'],ncol=6,loc=1)

u_LQR = np.vstack((u_LQR,[0.0,0.0]))
print Cost(5,x_LQR,u_LQR)
print .5*np.dot(np.dot(np.asarray([1.,1.0,0.0,0.0,0.0,0.0]).T,P[0]),
                np.asarray([1.,1.0,0.0,0.0,0.0,0.0]))

"""
Problem 4
"""

H = np.identity(6)



delta = 0.1

Ad = la.expm(delta*A)
Bd = np.dot(np.dot(np.dot(Ad,(np.identity(6) - sla.inv(Ad))),sla.pinv(A)),B)
Fd = Bd

mu0 = [[0.],[0.0],[0.],[0.],[0.],[0.]]
Sigma0 = [[ 0.1, 0.,0,0,0,0 ],[ 0., 0.1,0,0,0,0 ], [ 0., 0.,0.1,0,0,0 ], 
          [ 0., 0.,0,0.1,0,0 ], [ 0., 0.,0,0,0.1,0 ], [ 0., 0.,0,0,0,0.1 ]]
#Sigma0= sci.random.rand(6,6)

def kalman_filter(x0,P0,A,B,u,F,Q,H,R,t_,z_):
    
    x_ = [x0]; P_ = [P0]
    for t in range(len(t_)-1):
        xt_ = np.dot(A,x_[-1]) + np.dot(B,u[t])
        if isinstance(Q,float): 
            Pt_ = np.dot(A,np.dot(P_[-1],A.T)) + Q*np.dot(F,np.transpose(F))
        else:
            Pt_ = np.dot(A,np.dot(P_[-1],np.transpose(A))) + Q
        
        S = np.dot(np.dot(H, Pt_), H.T) + R
        if not S.shape: 
            K = (1.0/S)*np.dot(Pt_,H.T)
            xt = xt_ + (z_[t] - np.dot(H, xt_))*K.reshape(xt_.shape)
        else:
            K = np.dot(np.dot(Pt_, np.transpose(H)), la.inv(S))
            xt = xt_ + np.dot(K,(z_[t] - np.dot(H, xt_)))
        Pt = np.dot(np.identity(np.shape(Pt_)[0]) - np.dot(K,H),Pt_)
        x_.append(xt)
        P_.append(Pt)
    return np.asarray(x_),np.asarray(P_)

C=[[1, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0]]
D = np.asarray(0.0)
Qk= sci.random.rand(6,6)
QT = 100.0*np.dot(Qk,Qk.transpose())
Rk= sci.random.rand(6,6)
RT = 1.0*np.dot(Rk,Rk.transpose())

U,L,V = sla.svd(QT, full_matrices=True)
def fRand(t,X,U,A,B):
    return np.dot(A,X) + np.dot(B,U) + np.dot(Fd,np.dot(L,(
            np.random.multivariate_normal(np.zeros(6), np.identity(6), 2).T)))

def hRand(t,X,U,C,D):
    return np.dot(C,X) + np.dot(la.sqrtm(RT),
                  np.random.multivariate_normal(np.zeros(6), RT, 1).T)
    
dt = .01
def simRan(f,h,t,X0,U0,Y0,A,B,C,D):
    j,t_,X_,U_,Y_ = 0,[0],[X0],[U0],[Y0]
    while j*dt < t:
        t_.append((j+1)*dt)
        X_.append(X_[-1] + dt*f(j*dt,X_[-1],U_[-1],A,B))
        Y_.append( h(j*dt,X_[0],U_[0],C,D))
        U_.append(U_[-1])
        j += 1
    return np.asarray(t_), (X_), np.asarray( U_), np.asarray(Y_)

x0Test = [[2.0],[4.0], [0], [1], [1], [0]]
u0Test = [[1.0],[1.0]]
y0Test = hRand(0,x0Test,u0Test,H,0)
tfTest = 5
tT,xT,uT,zT = simRan(fRand,hRand,tfTest,x0Test,u0Test,y0Test,A,B,H,0)

#np.asarray asfortranarray

xHat, sigmaHat = kalman_filter(mu0,Sigma0,Ad,Bd,uT,Fd,QT,H,RT,tT,zT)

fig = plt.pyplot.figure()

plt.pyplot.plot(xHat[:,0])
#plt.pyplot.plot(zT[:])
plt.pyplot.title('Angular Position Estimate vs Observation')
plt.pyplot.ylabel('Angle (rad)')
plt.pyplot.xlabel('Time Step')
plt.pyplot.legend('Estimate', 'Observation')

fig = plt.pyplot.figure()
plt.pyplot.plot(xHat[:,0],label='Pos. Est.')
plt.pyplot.plot(xHat[:,1],label='Veloc. Est.')
plt.pyplot.title('Angular Position and Velocity Estimates')
plt.pyplot.ylabel('Angle (rad), Angular Velocity (rad/sec)')
plt.pyplot.xlabel('Time Step')
plt.pyplot.legend()

#fig = plt.pyplot.figure()
#plt.pyplot.psd(xHat[:,0], 512, 1 / dt)


plt.pyplot.rcParams['figure.figsize'] = (10, 8)
n_iter = 50
sz = (n_iter,) # size of array
x = 0
z = np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)

Qk= sci.random.rand(6,6)
Q = 100.0*np.dot(Qk,Qk.transpose()) # process variance

# allocate space for arrays
xhat=np.zeros(sz)      # a posteri estimate of x
P=np.zeros(sz)         # a posteri error estimate
xhatminus=np.zeros(sz) # a priori estimate of x
Pminus=np.zeros(sz)    # a priori error estimate
K=np.zeros(sz)         # gain or blending factor

R = 0.1**2 # estimate of measurement variance, change to see effect

# intial guesses
xhat[0] = 0.0
P[0] = 1.0

for k in range(1,n_iter):
    # time update
    xhatminus[k] = xhat[k-1]
    Pminus[k] = P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
    P[k] = (1-K[k])*Pminus[k]

plt.pyplot.figure()
plt.pyplot.plot(z,'k+',label='noisy measurements')
plt.pyplot.plot(xhat,'b-',label='a posteri estimate')
plt.pyplot.axhline(x,color='g',label='truth value')
plt.pyplot.legend()
plt.pyplot.title('Estimate vs. iteration step', fontweight='bold')
plt.pyplot.xlabel('Iteration')
plt.pyplot.ylabel('Voltage')

plt.pyplot.figure()
valid_iter = range(1,n_iter) # Pminus not valid at step 0
plt.pyplot.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
plt.pyplot.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', fontweight='bold')
plt.pyplot.xlabel('Iteration')
plt.pyplot.ylabel('$(Voltage)^2$')
plt.pyplot.setp(plt.gca(),'ylim',[0,.01])
plt.pyplot.show()

