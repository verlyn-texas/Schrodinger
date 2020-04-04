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


def create_field(x_extent, t_extent):

    # Create simulation field
    # t, x, (complex, potential, boundary)
    # boundary is 1 if true and 0 if false
    # treat -1 as an infinite potential
    field = np.zeros((t_extent, x_extent, 3), dtype=complex)

    return field


def set_initial_values(field, particles, real, imag):

    t_extent = field.shape[0]
    x_extent = field.shape[1]
    for t in range(t_extent):
        for x in range(x_extent):
            field[t, x, psi_index] = complex(real, imag)
        normalize(field, t, particles)


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


def set_free_particle_boundary(field, h, m, a, k):
    t_extent = field.shape[0]
    x_extent = field.shape[1]
    x_offset = 300
    t = 1
    for x in range(x_extent):
        field[t, x, psi_index] = get_free_psi(x - x_offset, t, h, m, a, k)
        field[t, x, boundary_index] = complex(1, 0)
        field[t, x, potential_index] = complex(0, 0)
    t = 2
    for x in range(x_extent):
        field[t, x, psi_index] = get_free_psi(x - x_offset, t, h, m, a, k)
        field[t, x, boundary_index] = complex(0, 0)
        field[t, x, potential_index] = complex(0, 0)

    for t in range(t_extent):
        field[t, 0, psi_index] = complex(0, 0)
        field[t, 1, psi_index] = complex(0, 0)


def fit(field, h, m):

    t_extent = field.shape[0]
    x_extent = field.shape[1]

    for t in range(2, t_extent):
        for x in range(2, x_extent):
            if field[t, x, boundary_index] != complex(1, 0):
                psi = get_best_psi(field, x, t, h, m)
                field[t, x, psi_index] = psi
        normalize(field, t, 1)

####################################################################################
# Internal Functions


def get_boundary_psi(width, n, x):
    real = math.sqrt(2 / width) * math.sin(n * math.pi * x / width)
    imaginary = 0
    value = complex(real, imaginary)
    return value


def get_free_psi(x, t, h, m, a, k):

    T = 2 * m * a * a / h
    A = 1 / (a ** 0.5 * (2 * math.pi) ** 0.25 * (1 + complex(0, 1) * t / T) ** 0.5)
    term_first = complex(0, 1) * T * (x / (2 * a)) ** 2 / t
    term_second = - ((complex(0, 1) * T / (4 * a * a * t)) * (x - h * k * t / m) ** 2) / (1 + complex(0, 1) * t / T)
    psi = A * cmath.exp(term_first) * cmath.exp(term_second)

    return psi


def get_best_psi(field, x, t, h, m):

    t1 = field[t - 1, x, psi_index]
    x1 = field[t, x - 1, psi_index]
    x2 = field[t, x - 2, psi_index]
    v = field[t, x, potential_index]

    if v == complex(-1, 0):
        psi = complex(0, 0)
    else:
        psi_num = complex(0, 1) * h * t1 + (h * h / m) * (x1 - x2/2)
        psi_den = complex(0, 1) * h + h * h / (2 * m) - v
        psi = psi_num / psi_den

    return psi


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

    t_range = [1, 2, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
    # t_range = [1, 2]

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
        # plt.plot(real_list)
        # plt.plot(im_list)
        plt.legend(['prob', 'real', 'imag'])
        plt.title(f'T={t}')
        plt.show()


####################################################################################
# Main

def main():

    # Control Parameters

    particles = 1
    low_energy = 1
    high_energy = 2
    wall_height = -1
    h = 1.0
    m = 400
    init_real = 0.1
    init_imag = -0.1
    a = 50
    v = 2
    k = m * v / h
    print(f'k:{k}')

    # Setup

    field = create_field(1000, 1000)
    set_initial_values(field, particles, init_real, init_imag)

    # Two Wells and Trench
    # set_potential_two_wells(field, wall_height)
    # set_trench_potential(field, init_real, init_imag)
    # set_boundaries(field, low_energy, high_energy)

    # Free particle
    set_free_particle_boundary(field, h, m, a, k)
    fit(field, h, m)

    # Analysis

    plotSolution(field)
    three_d_plot(field)


main()