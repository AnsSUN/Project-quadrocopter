# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 21:22:43 2018

@author: anshu
"""




import numpy as np
import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D


def quad_nlh(h, v, theta):
    h_dot=h
    v_dot=v
    theta_dot=theta
    return h_dot, v_dot, theta_dot


def quad_nlf(h, v, theta, u=[]):
    m=1.
    g=9.81 
    I=1.
    h_ddot = (u[0, :]/m)*np.sin(theta) # theta in radians
    v_ddot = -g + (u[0, :]/m)*np.cos(theta)
    theta_ddot = u[1, :]/I
    return h_ddot, v_ddot, theta_ddot


dt = 0.01
stepCnt = 10000
t = np.arange(0,100,.1)
omg=0.0025 # rad/sec
tst=100
oneele=9.81+np.sin(2*t*np.pi*omg) # max value input
ut=np.zeros(shape=(2,np.size(t) + 1))

for k in range(np.size(oneele)):
    ut[0,k+1]= oneele[k]
   

# Need one more for the initial values
hsh = np.empty((stepCnt + 1,))
vsh = np.empty((stepCnt + 1,))
thetash = np.empty((stepCnt + 1,))
hsf = np.empty((stepCnt + 1,))
vsf = np.empty((stepCnt + 1,))
thetasf = np.empty((stepCnt + 1,))


# Setting initial values
hsh[0], vsh[0], thetash[0] = (1., 1., 1.57)
u=np.matrix([[9.81],[0]]) # part 1 b


# Stepping through "time".
for i in range(stepCnt):
    # Derivatives of the X, Y, Z state
    h_dot, v_dot, theta_dot= quad_nlh(hsh[i], vsh[i], thetash[i])
    hsh[i + 1] = hsh[i] + (h_dot * dt)
    vsh[i + 1] = vsh[i] + (v_dot * dt)
    thetash[i + 1] = thetash[i] + (theta_dot * dt)
for j in range(stepCnt):
    h_ddot, v_ddot, theta_ddot = quad_nlf(hsh[j], vsh[j], thetash[j], ut)
    hsf[j + 1] = hsf[j] + (h_ddot * dt)
    vsf[j + 1] = vsf[j] + (v_ddot * dt)
    thetasf[j + 1] = thetasf[j] + (theta_ddot * dt)

fig = plt.pyplot.figure()
ax = fig.gca(projection='3d')

ax.plot(hsf, vsf, thetasf, lw=0.5)
ax.set_xlabel("Horizontal position")
ax.set_ylabel("Vertical position")
ax.set_zlabel("theta position")
ax.set_title("Quadrotor model at theta= 90deg, u=[1, 1]'")

fig = plt.pyplot.figure()
plt.pyplot.plot(hsf, vsf, 'r')
plt.pyplot.xlabel("Horizontal position")
plt.pyplot.ylabel("Vertical position")



plt.pyplot.show()


