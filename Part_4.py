import numpy as np
import _547 as lst
import scipy as sp
import matplotlib.pyplot as plt
import control
import matplotlib.animation as animation
g,m,I = 9.81,1.,1.
np.set_printoptions(precision=2)

#Part1 (a)
def generate_dynamics(g,m,I):
	def f(t,x,u):
		h,v,th,hdot,vdot,thdot = x
		#print u
		xdot = np.array([hdot,vdot,thdot,u[0]/m*np.sin(th), \
			-g+u[0]/m*np.cos(th),u[1]/I])
		return xdot
	return f

def h(t,x,u):
	h,v,th,hdot,vdot,thdot = x
	return np.array([h,v])

def u_0(t):
	m = 1
	g = 9.81
	return np.array([m*g,0])

def u_sin(t):
	m = 1
	g = 9.81
	u1 = m*g+np.sin(2*t*np.pi*0.5)
	u2 = 0
	return np.array([u1,u2])

f_1 = generate_dynamics(g,m,I)
x_0 = np.array([0.1,0.1,0,0,0,0])

tf = 15


#PART2:Stabilization
#(a) linearize about the point
A = np.matrix([[0,0,0,1,0,0],
				[0,0,0,0,1,0],
				[0,0,0,0,0,1],
				[0,0,g,0,0,0],
				[0,0,0,0,0,0],
				[0,0,0,0,0,0]])
B = np.matrix([[0,0],
				[0,0],
				[0,0],
				[0,0],
				[1/m,0],
				[0,1/I]])
#(b) check if (A,B) controllable
print lst.controllable(A,B)

#l = labmda that sufficiently large
l = 5
A_stable = -l*np.eye(6)-A
P = []
if lst.controllable(A_stable,B):
	W = control.lyap(A_stable,np.dot(B,B.T))
	P_0 = np.linalg.inv(W)
	#print P_0
	K_0 = 0.5*np.dot(B.T,P_0)
	w,v = np.linalg.eig(A_stable-np.dot(B,K_0))
	#print w

def u_Stable_feedback_controller(x):
	u = np.dot(-K_0,x)
	u = np.ravel(u)
	return u


def PSD(size,sqrt=False):
	H = np.random.rand(size,size)
	d,u = np.linalg.eig(H+H.T)
	S = np.dot(u,np.dot(np.diag(np.sqrt(d*np.sign(d))),u.T))
	if sqrt:
		return np.dot(S.T,S),S
	else:
		return np.dot(S.T,S)

Q = PSD(6)
R = PSD(2)

def lqrCT(A,B,Q,R,T,dt=1e-4):
	P = []
	Pt=Q
	P.append(Pt)
	Kt=-np.dot(np.linalg.inv(R),np.dot(B.T,Pt))
	k = T/dt
	while k>=1:
		Pt_1=dt*(np.dot(A.T,Pt)+np.dot(Pt,A)-np.dot(Pt,np.dot(B,np.dot(np.linalg.inv(R),np.dot(B.T,Pt))))+Q)+Pt
		kt = -np.dot(np.linalg.inv(R),np.dot(B.T,Pt_1))
		P.append(Pt_1)
		Pt = Pt_1
		k = k-1
	#return Pt
	return P

def u_lqr(x):
	u = np.dot(K_3,x)
	u = np.ravel(u)
	return u
x_0_ = np.array([0.1,0.1,0,0,0,0])


#Part 4
A_closed = A-np.dot(B,K_0)
w,v = np.linalg.eig(A_closed)
#print w
F_t = np.zeros((6,6))
Sampling_interval = 0.01

A_d = sp.linalg.expm(A_closed*Sampling_interval)
#print A_d
w_,v = np.linalg.eig(A_d)



dt = 1e-4
t_int = 0
while t_int<Sampling_interval:
	matExp = sp.linalg.expm(t_int*A_closed)
	F_t = dt*matExp
	t_int+=dt
#F_t = np.dot(F_t,B)
F_t = np.matrix([[0.1,0.],[0.,0.1],[0.1,0.],[0.,0.1],[0.1,0.],[0.,0.1]])
print F_t
C_d = np.matrix([[1,0,0,0,0,0],
				 [0,1,0,0,0,0]])
H_t = np.eye(2)
sig_0 = np.diag(np.array([0.1,0.1,0.1,0.1,0.1,0.1]))
#sig_0 = 10*PSD(6)
Q = 0.1*np.eye(2)
R = 0.2*np.eye(2)
wt = np.random.multivariate_normal([0,0],Q,2000)
#print wt
vt = np.random.multivariate_normal([0,0],R,2000)
#print vt
x_0_ = np.array([10,10,np.pi,10,10,np.pi/6])

x_4 = []
y_4 = []
y_4_noise = []
tf = 2000
t = 0
while t<tf:
	if t==0:
		x_4.append(x_0_)
	else:
		x_4_ = np.dot(A_d,np.array(x_4[-1]).T)+np.dot(F_t,np.array(wt[t]).T)
		x_4.append(np.ravel(x_4_))
	y_4_noise_ = np.dot(C_d,np.array(x_4[-1]).T)+np.dot(H_t,np.array(vt[t]).T)
	y_4_ = np.dot(C_d,np.array(x_4[-1]).T)
	t+=1
	y_4_noise.append(np.ravel(y_4_noise_))
	y_4.append(np.ravel(y_4_))
y_4 = np.array(y_4)
y_4_noise=np.array(y_4_noise)
x_4 = np.array(x_4)

def kalman_filter(x0,P0,A,B,C,F,Q,H,R,tf,z_,u=None):
	x_hat = []
	P_ = []
	t_ = []
	y_ = []
	t = 0
	x_hat_t_t_1 = x0
	sig_t_t_1 = P0

	x_hat_t_t = np.array([x_hat_t_t_1]).T+sig_t_t_1*C.T*np.linalg.inv(C*sig_t_t_1*C.T+H*R*H.T)*(np.array([z_[0]]).T-C*np.array([x_hat_t_t_1]).T)
	y_hat_t_t = np.dot(C_d,x_hat_t_t)
	y_.append(np.ravel(y_hat_t_t.T))
	#print x_hat_t_t
	sig_t_t = sig_t_t_1- sig_t_t_1*C.T*np.linalg.inv(C*sig_t_t_1*C.T+H*R*H.T)*C*sig_t_t_1
	x_hat.append(np.ravel(x_hat_t_t.T))
	P_.append(sig_t_t)
	x_hat_t_1_t_1 = x_hat_t_t
	sig_t_1_t_1 = sig_t_t
	t_.append(t)
	t+=1
	while t<tf: 
		x_hat_t_t_1 = np.dot(A,x_hat_t_1_t_1)
		sig_t_t_1 = np.dot(A,np.dot(sig_t_1_t_1,A.T)) + np.dot(F,np.dot(Q,F.T))
		
		x_hat_t_t = x_hat_t_t_1+sig_t_t_1*C.T*np.linalg.inv(C*sig_t_t_1*C.T+H*R*H.T)*(np.array([z_[t]]).T-C*x_hat_t_t_1)
		sig_t_t = sig_t_t_1- sig_t_t_1*C.T*np.linalg.inv(C*sig_t_t_1*C.T+H*R*H.T)*C*sig_t_t_1
		
		y_hat_t_t = np.dot(C_d,x_hat_t_t)
		y_.append(np.ravel(y_hat_t_t.T))

		x_hat.append(np.ravel(x_hat_t_t.T))
		P_.append(sig_t_t)
		x_hat_t_1_t_1 = x_hat_t_t
		sig_t_1_t_1 = sig_t_t
		t_.append(t)
		t+=1

	return x_hat, P_, y_,t_
tf = 2000
x_hat, P_,y_t_estimate, t_ = kalman_filter(x_0_,sig_0,A_d,B,C_d,F_t,Q,H_t,R,tf,y_4_noise)
t_ = [i*0.01 for i in t_]
x_hat = np.array(x_hat)
fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,figsize=(16,10))
ax1.plot(t_[:],x_hat[:,0],lw = 2,label='estimated horizontal position')
ax1.plot(t_[:],x_4[:,0],lw = 2,label='horizontal position')
ax1.legend(loc=1,fontsize=12)
ax1.set_xlabel('t')
ax1.set_ylabel('h')
ax2.plot(t_[:],x_hat[:,1],lw = 2,label='estimated vertical position')
ax2.plot(t_[:],x_4[:,1],lw = 2,label='vertical position')
ax2.legend(loc=1,fontsize=12)
ax2.set_xlabel('t')
ax2.set_ylabel('v')
ax3.plot(t_[:],x_hat[:,2],lw = 2,label='estimated rotation')
ax3.plot(t_[:],x_4[:,2],lw = 2,label='rotation')
ax3.legend(loc=1,fontsize=12)
ax3.set_xlabel('t')
ax3.set_ylabel('theta')
ax4.plot(t_[:],x_hat[:,3],lw = 2,label='estimated horizontal velocity')
ax4.plot(t_[:],x_4[:,3],lw = 2,label='h velocity')
ax4.legend(loc=1,fontsize=12)
ax4.set_xlabel('t')
ax4.set_ylabel('h dot')
ax5.plot(t_[:],x_hat[:,4],lw = 2,label='estimated vertical velocity')
ax5.plot(t_[:],x_4[:,4],lw = 2,label = 'vertical velocity')
ax5.legend(loc=1,fontsize=12)
ax5.set_xlabel('t')
ax5.set_ylabel('v dot')
ax6.plot(t_[:],x_hat[:,5],lw = 2,label='estimated rotation velocity')
ax6.plot(t_[:],x_4[:,5],lw = 2,label='rotation velocity')
ax6.legend(loc=1,fontsize=12)
ax6.set_xlabel('t')
ax6.set_ylabel('theta dot')
plt.show()

y_t_estimate = np.array(y_t_estimate)

fig, (ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,figsize=(16,10))
ax1.plot(t_[:],y_4[:,0],lw=2,label='observed horizontal position')
ax1.legend(loc=1,fontsize=12)
ax1.set_xlabel('t')
ax1.set_ylabel('y_1')
ax2.plot(t_[:],y_4[:,1],lw=2,label='observed vertical position')
ax2.legend(loc=1,fontsize=12)
ax2.set_xlabel('t')
ax2.set_ylabel('y_2')
ax3.plot(t_[:],y_4_noise[:,0],lw=2,label='noisy observed horizontal position')
ax3.legend(loc=1,fontsize=12)
ax3.set_xlabel('t')
ax3.set_ylabel('y_1')
ax4.plot(t_[:],y_4_noise[:,1],lw=2,label='noisy observed vertical position')
ax4.legend(loc=1,fontsize=12)
ax4.set_xlabel('t')
ax4.set_ylabel('y_2')
ax5.plot(t_[:],y_t_estimate[:,0],lw=2,label='estimated observed horizontal position')
ax5.legend(loc=1,fontsize=12)
ax5.set_xlabel('t')
ax5.set_ylabel('y_1')
ax6.plot(t_[:],y_t_estimate[:,1],lw=2,label='estimated observed vertical position')
ax6.legend(loc=1,fontsize=12)
ax6.set_xlabel('t')
ax6.set_ylabel('y_2')
plt.show()






