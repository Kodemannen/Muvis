# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import murstein
from matplotlib import animation

a, b, c = 5., 10., 20.      # cm
density = 2./(10**3)        # kg/cm^3
total_mass = density*a*b*c
Jb = np.zeros((3,3))
Jb[0,0] = total_mass*(b**2 + c**2) / 12.      # Jxx
Jb[1,1] = total_mass*(a**2 + c**2) / 12.      # Jyy
Jb[2,2] = total_mass*(a**2 + b**2) / 12.      # Jzz

total_time = 10        # s
fps = 7
dt = 1e-5    # s
m = int(round(1./(dt*fps)))     # m is a variable that ensures the time count and angular velocity in the animation is correct
N = int(total_time/dt + 1)
t = np.linspace(0, total_time, N)
print("dt = %s"%dt)
print("N = %i"%N)
print("m = %i"%m)


# ############################
# # Plotter initiell konfig: #
# ############################
# fig0 = plt.figure()
# ax0 = Axes3D(fig0)
# brick0 = murstein.make_brick(a,b,c)
# murstein.plot_brick(ax0, brick0)
# k = c*3./4
# ax0.set_xlim(-k,k)
# ax0.set_ylim(-k,k)
# ax0.set_zlim(-k,k)
# ax0.scatter(0,0,0)   # makes origo
# ax0.set_xlabel("x")
# ax0.set_ylabel("y")
# ax0.set_zlabel("z")
# plt.title("Config at t=0")
# plt.show()
# import sys
# sys.exit()


##############################################################
# Setter initialbetingelsen, dvs. vinkelhastighet ved t = 0: #
##############################################################
K0 = 2*np.pi**2 * Jb[2,2]   # kinetisk energi som tilsvarer 1 hz rotasjon rundt z-aksen
pure_w1 = np.sqrt(2*K0/Jb[0,0])
pure_w2 = np.sqrt(2*K0/Jb[1,1])
pure_w3 = 2*np.pi
frac = 10.   # fraction of pure rotation on main axis that will be the perturbation
# Velger de tre rotasjonene nær de tre hovedaksene:
omegas = [
    [pure_w1*np.sqrt(1-1./frac**2), pure_w2/frac, 0],
    [pure_w1/frac, pure_w2*np.sqrt(1-1./frac**2), 0],
    [0, pure_w2/frac, pure_w3*np.sqrt(1-1./frac**2)]
]

# print(omegas)

# import sys
# sys.exit()
# axises = ["x","y","z"]

for axis in range(1,4):

    mainRotAxis = axis
    print("Integrating for axis = %s" % axis)
    index = axis-1
    theta = np.zeros((N, 3))
    theta_deriv = np.zeros((N,3))

    omega = np.zeros((N, 3))
    alpha = np.zeros((N, 3))

    initial_omega = np.array(omegas[index])
    omega[0] = initial_omega
    energy = np.sum(np.dot(Jb,omega[0]**2))*0.5
    print("Energy at t=0: ", energy)

    ###############################
    # Integrerer eulerlikningene: #
    # Metode: Euler chromer       #
    ###############################
    constx = (Jb[2,2]-Jb[1,1])/float(Jb[0,0])
    consty = (Jb[0,0]-Jb[2,2])/float(Jb[1,1])
    constz = (Jb[1,1]-Jb[0,0])/float(Jb[2,2])
    for i in range(N-1):
        alpha[i,0] = - omega[i,1]*omega[i,2]*constx
        alpha[i,1] = - omega[i,0]*omega[i,2]*consty
        alpha[i,2] = - omega[i,0]*omega[i,1]*constz

        omega[i+1] = omega[i] + alpha[i]*dt
        theta[i+1] = omega[i+1]*dt

        # D_mat = murstein.D_matrix(theta[i])
        # theta_deriv[i] = np.dot(D_mat, omega[i])
        # theta[i+1] = theta[i] + theta_deriv[i]*dt

        # #for i in range(N-1):
        # D_mat = murstein.D_matrix(theta[i])
        # theta_deriv[i] = np.dot(D_mat, initial_omega)
        # theta[i+1] = theta[i] + theta_deriv[i]*dt


        ###########################################################
        # Kode som printer energien for å sjekke at den er bevart:#
        ###########################################################
        # if i%1000000 == 0:
        #     energy = np.sum(np.dot(Jb,omega[i+1]**2))*0.5
        #     print("Energy = %.2f" % energy, "i = %s" % i)
    energy = np.sum(np.dot(Jb,omega[-1]**2))*0.5
    print("Energy at t=%s: " % total_time, energy)
    print("Done integrating")
    # import sys
    # sys.exit("asd")
    print("Plotting..")
    ########################################
    # Plotter rotasjonsvektoren over tid:  #
    ########################################
    fig = plt.figure()
    ax = Axes3D(fig)
    n = 2
    for i in range(0,N,m):
        vecx = np.linspace(0,omega[i,0], n)
        vecy = np.linspace(0,omega[i,1], n)
        vecz = np.linspace(0,omega[i,2], n)

        if i == 0:
            ax.plot(vecx, vecy, vecz, label = "Simulation time=%.1fs" % total_time)
        else:
            ax.plot(vecx, vecy, vecz)
    ax.legend()
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    plt.title("Main rotation axis = %s" % axis)
    plt.savefig("axisplot %s" % axis)
    #plt.show()


    #########################
    # Simulerer rotasjonen: #
    #########################
    print("Simulating rotation:")
    brick = murstein.make_brick(a,b,c)
    bricks = [brick]
    for i in range(0,N,m):
        angle1,angle2,angle3 = np.sum(theta[i:i+m,:],axis=0)
        rotMat = murstein.euler_321_matrix(angle1, angle2, angle3)
        new_brick = murstein.rotate_brick(rotMat, brick)
        brick = new_brick
        bricks.append(brick)

    ######################################
    # Plotter rotasjonen til mursteinen: #
    ######################################
    fig2 = plt.figure()
    rect1 = fig2.add_subplot(1, 2, 1).get_position()
    ax = Axes3D(fig2, rect1)
    n = 2
    # for i in range(N):
    vecx = np.linspace(0,omega[0,0], n)
    vecy = np.linspace(0,omega[0,1], n)
    vecz = np.linspace(0,omega[0,2], n)
    ax.plot(vecx, vecy, vecz)

    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(-10,10)
    #plt.title("Main rotation axis = %i" % mainRotAxis)
    #plt.savefig("axisplot %i" % mainRotAxis)


    #fig2 = plt.figure()
    rect2 = fig2.add_subplot(1, 2, 2).get_position()
    ax2 = Axes3D(fig2, rect2)
    k = c*3./4
    ax2.set_xlim(-k,k)
    ax2.set_ylim(-k,k)
    ax2.set_zlim(-k,k)
    ax2.scatter(0,0,0)   # makes origo
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")



    def update_data(i):
        # i = i*m
        if i > N-1:
            i = N-1
        if axis != 2:
            ax.clear()
        vecx = np.linspace(0,omega[i*m,0], n)
        vecy = np.linspace(0,omega[i*m,1], n)
        vecz = np.linspace(0,omega[i*m,2], n)

        ax.plot(vecx,vecy,vecz)
        ax.scatter(0,0,0)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_zlabel("z (cm)")
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_zlim(-5,5)
        ax.legend(["Close to axis: ", "$\omega(t_0)$ = [%.1f, %.1f, %.1f]" % tuple(initial_omega)])

        ax2.clear()

        ax2.set_xlim(-k,k)
        ax2.set_ylim(-k,k)
        ax2.set_zlim(-k,k)
        ax2.scatter(0,0,0)   # makes origo
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.legend(["Time = %.2f s" % t[i*m]])
        #plt.title("$\omega(t_0)$ = [%.1f, %.1f, %.1f]" % tuple(initial_omega))
        murstein.plot_brick(ax2, bricks[i])

    print("Animating..")
    ani = animation.FuncAnimation(fig2, update_data , int(round(N/float(m))))
    ani.save('mainRotAxis=%s.gif' % axis, writer='imagemagick', fps = fps)
    print("Done.")
    #plt.show()
