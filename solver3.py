import math
import cmath
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


# indexes
psi_index = 0
potential_index = 1
boundary_index = 2

# pixel conversions
# x_map = 9.45 * 10 ** 11  # for use with photons
x_map = 9.45E13  # for use with slow electrons
t_map = 9.45E15


####################################################################################
# Setup - General


def create_field(x_extent, t_extent):

    # Create simulation field
    # t, x, (complex, potential, boundary)
    # boundary is 1 if true and 0 if false
    # treat -1 as an infinite potential
    field = np.zeros((t_extent, x_extent, 3), dtype=complex)
    for x in range(x_extent):
        for t in range(t_extent):
            field[t, x, psi_index] = complex(0, 0)
            field[t, x, boundary_index] = complex(0, 0)
            field[t, x, potential_index] = complex(0, 0)

    return field


####################################################################################
# Setup - Free Particle


def set_free_particle_potential(field):
    t_extent = field.shape[0]
    x_extent = field.shape[1]

    for x in range(0, 5):
        for t in range(t_extent):
            field[t, x, psi_index] = complex(0, 0)
            field[t, x, boundary_index] = complex(1, 0)
            field[t, x, potential_index] = complex(-1, 0)

    for x in range(x_extent - 5, x_extent):
        for t in range(t_extent):
            field[t, x, psi_index] = complex(0, 0)
            field[t, x, boundary_index] = complex(1, 0)
            field[t, x, potential_index] = complex(-1, 0)


def set_free_particle_boundary(field, k, a, free_particle_offset):

    x_extent = field.shape[1]

    t = 0
    for x in range(x_extent):
        field[t, x, psi_index] = get_free_psi(x - free_particle_offset, a, k)
        field[t, x, boundary_index] = complex(0, 0)  # need to be able to normalize so use 0,0
        field[t, x, potential_index] = complex(0, 0)
    normalize(field, t, 1)


def get_free_psi_old(x, t, h_bar, m, a, k):

    x_phy = x / x_map
    t_phy = t / t_map

    if x == 500:
        print('pause')

    T = 2 * m * a * a / h_bar
    T = 1
    A = 1 / (a ** 0.5 * (2 * math.pi) ** 0.25 * (1 + complex(0, 1) * t_phy / T) ** 0.5)
    term_first = complex(0, 1) * T * (x_phy / (2 * a)) ** 2 / t_phy
    term_second = - ((complex(0, 1) * T / (4 * a * a * t_phy)) * (x_phy - h_bar * k * t_phy / m) ** 2) / (1 + complex(0, 1) * t_phy / T)
    psi = A * cmath.exp(term_first) * cmath.exp(term_second)

    return psi


def get_free_psi(x_sim, a, k):


    # provides a Gaussian wave packet with average momentum h_bar * k with a width of '2 * a'

    x = x_sim / x_map

    A = 1 / (a ** 0.5 * (2 * math.pi) ** 0.25)
    term_first = complex(0, 1) * k * x
    term_second = - x * x / (4 * a * a)

    psi = A * cmath.exp(term_first) * cmath.exp(term_second)

    return psi


####################################################################################
# Setup - Two Atoms


def set_potential_two_wells(field, height):

    t_extent = field.shape[0]

    for x in range(0, 200):
        for t in range(t_extent):
            field[t, x, potential_index] = complex(height, 0)
            field[t, x, psi_index] = complex(0, 0)
            field[t, x, boundary_index] = complex(1, 0)

    for x in range(401, 600):
        for t in range(t_extent):
            field[t, x, potential_index] = complex(height, 0)
            field[t, x, psi_index] = complex(0, 0)
            field[t, x, boundary_index] = complex(1, 0)

    for x in range(801, 1000):
        for t in range(t_extent):
            field[t, x, potential_index] = complex(height, 0)
            field[t, x, psi_index] = complex(0, 0)
            field[t, x, boundary_index] = complex(1, 0)


def set_trench_potential(field, real, imag):
    x_start = 401
    x_end = 599
    t_start = 200
    t_end = 800
    potential = 0.0

    for t in range(t_start, t_end+1):
        for x in range(x_start, x_end+1):
            field[t, x, potential_index] = complex(potential, 0)
            field[t, x, psi_index] = complex(real, imag)
            field[t, x, boundary_index] = complex(0, 0)


def set_boundaries(field, low_energy, high_energy):

    x_extent = field.shape[1]

    t = 0
    for x in range(x_extent):
        field[t, x, psi_index] = complex(0, 0)
        field[t, x, boundary_index] = complex(1, 0)

    prob_total = 0
    for x in range(200, 401):
        relative_x = x - 200
        value = get_boundary_psi(200, low_energy, relative_x)
        prob_total += (value * complex.conjugate(value)).real
        field[t, x, psi_index] = value
        field[t, x, boundary_index] = complex(1, 0)
    print(f'Probability Left: {prob_total} at t={t}')

    prob_total = 0
    for x in range(600, 801):
        relative_x = x - 600
        value = get_boundary_psi(200, high_energy, relative_x)
        prob_total += (value * complex.conjugate(value)).real
        field[t, x, psi_index] = value
        field[t, x, boundary_index] = complex(1, 0)
    print(f'Probability Right: {prob_total} at t={t}')

    # t = 1
    # for x in range(x_extent):
    #     field[t, x, psi_index] = complex(0, 0)
    #     field[t, x, boundary_index] = complex(1, 0)
    #
    # prob_total = 0
    # for x in range(200, 401):
    #     relative_x = x - 200
    #     value = get_boundary_psi(200, low_energy, relative_x)
    #     prob_total += (value * complex.conjugate(value)).real
    #     field[t, x, psi_index] = value
    #     field[t, x, boundary_index] = complex(1, 0)
    # print(f'Probability Left: {prob_total} at t={t}')
    #
    # prob_total = 0
    # for x in range(600, 801):
    #     relative_x = x - 600
    #     value = get_boundary_psi(200, high_energy, relative_x)
    #     prob_total += (value * complex.conjugate(value)).real
    #     field[t, x, psi_index] = value
    #     field[t, x, boundary_index] = complex(1, 0)
    # print(f'Probability Right: {prob_total} at t={t}')


def get_boundary_psi(width, n, x):
    real = math.sqrt(2 / width) * math.sin(n * math.pi * x / width)
    imaginary = 0
    value = complex(real, imaginary)
    return value


####################################################################################
# Simulator


def fit(field, h_bar, m):

    t_extent = field.shape[0]
    x_extent = field.shape[1]

    for t in range(1, t_extent):
        for x in range(x_extent):
            if field[t, x, boundary_index] != complex(1, 0):
                psi = get_best_psi(field, x, t, h_bar, m)
                field[t, x, psi_index] = psi
        normalize(field, t, 1)


def get_best_psi(field, x, t, h_bar, m):

    # todo - not progressing in time

    t1 = field[t - 1, x, psi_index]
    x1 = field[t - 1, x - 1, psi_index]
    x2 = field[t - 1, x - 2, psi_index]
    v = field[t, x, potential_index]

    if v == complex(-1, 0):
        psi = complex(0, 0)
    else:
        psi_num = complex(0, 1) * h_bar * t1 + (h_bar * h_bar / m) * (x1 - x2 / 2)
        psi_den = complex(0, 1) * h_bar + h_bar * h_bar / (2 * m) - v
        psi = psi_num / psi_den

    return psi


####################################################################################
# Internal Functions


def normalize(field, t, particles):
    x_extent = field.shape[1]
    sum_prob = 0
    new_sum_prob = 0
    for x in range(x_extent):
        if field[t, x, boundary_index] != complex(1, 0):
            sum_prob += (field[t, x, psi_index] * complex.conjugate(field[t, x, psi_index])).real
    for x in range(x_extent):
        if field[t, x, boundary_index] != complex(1, 0):
            field[t, x, psi_index] = math.sqrt(particles) * field[t, x, psi_index] / math.sqrt(sum_prob)
            new_sum_prob += (field[t, x, psi_index] * complex.conjugate(field[t, x, psi_index])).real
    # print(f'    t: {t}   Start Prob: {sum_prob}   New Prob: {new_sum_prob}')


####################################################################################
# Analysis Functions


def three_d_plot(field):
    t_extent = field.shape[0]
    x_extent = field.shape[1]

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(x_extent)
    T = np.arange(t_extent)
    X, T = np.meshgrid(X, T)
    R = np.sqrt(X ** 2 + T ** 2)
    Z = np.sin(R)
    # Z = np.zeros((t_extent, x_extent))

    for x in range(0, x_extent):
        for t in range(0, t_extent):
            prob = (field[t, x, psi_index] * complex.conjugate(field[t, x, psi_index])).real
            Z[t, x] = prob

    # Plot the surface.
    surf = ax.plot_surface(T, X, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0.0, 0.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.03f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plotSolution(field):
    t_extent = field.shape[0]
    x_extent = field.shape[1]

    img = np.zeros((t_extent, x_extent))

    for x in range(0, x_extent):
        for t in range(0, t_extent):
            prob = (field[t, x, psi_index] * complex.conjugate(field[t, x, psi_index])).real
            img[t, x] = prob

    plt.imshow(img, cmap='gray', vmin=0.0, vmax=0.02)
    plt.title('Probability')
    plt.show()

    t_range = [0, 1, 2, 3, 75, 100, 1499]
    # t_range = [0]

    for t in t_range:
        prob_list = []
        real_list = []
        im_list = []
        for x in range(0, x_extent):
            real_val = field[t, x, psi_index].real
            im_val = field[t, x, psi_index].imag
            prob = (field[t, x, psi_index] * complex.conjugate(field[t, x, psi_index])).real
            prob_list.append(prob)
            real_list.append(real_val)
            im_list.append(im_val)

        plt.plot(prob_list)
        plt.plot(real_list)
        plt.plot(im_list)
        plt.legend(['prob', 'real', 'imag'])
        plt.title(f'T={t}')
        plt.show()


####################################################################################
# Main

def main():

    # Control Parameters
    h_bar = 1.05E-34  # J s
    c = 3.00E8  # m/s

    # Free particle, photon
    free_particle_offset = 400  # pixels
    a = 2.82E-13  # m; radius of electron
    m = 9.11E-31  # kg
    v = 100  # m/s
    k = m * v / h_bar

    # Two Atoms
    particles = 1
    low_energy = 1
    high_energy = 2

    # Setup
    field = create_field(1500, 1500)

    # Free particle
    set_free_particle_potential(field)
    set_free_particle_boundary(field, k, a, free_particle_offset)
    fit(field, h_bar, m)

    # Two Wells and Trench
    # set_potential_two_wells(field, wall_height)
    # set_trench_potential(field, init_real, init_imag)
    # set_boundaries(field, low_energy, high_energy)
    # fit(field, h_bar, m)

    # Analysis

    plotSolution(field)
    three_d_plot(field)


main()