import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import random


# indexes
psi_index = 0
potential_index = 1
boundary_index = 2


def createField(x_extent, t_extent):
    # Create simulation field
    # t, x, (complex, potential, boundary)
    # boundary is 1 if true and 0 if false
    # treat -1 as an infinite potential
    field = np.zeros((t_extent, x_extent, 3), dtype=complex)

    return field


def setWellPotential(field, particles):
    # Create potential walls

    t_extent = field.shape[0]
    x_extent = field.shape[1]

    for x in range(0, 400):
        for t in range(t_extent):
            field[t, x, potential_index] = complex(1, 0)
            field[t, x, boundary_index] = complex(1, 0)

    for x in range(601, 1000):
        for t in range(t_extent):
            field[t, x, potential_index] = complex(1, 0)
            field[t, x, boundary_index] = complex(1, 0)

    # Set initial values in well
    for t in range(t_extent):
        for x in range(400, 600):
            r = random.uniform(0, 0.01)
            i = random.uniform(0, 0.01)
            field[t, x, psi_index] = complex(0.1, 0.1)
        normalize(field, t, particles)


def getState(width, n, x):
    real = math.sqrt(2 / width) * math.sin(n * math.pi * x / width)
    imaginary = 0
    value = complex(real, imaginary)
    return value


def setWellBoundary(field, energy):
    # Set boundary conditions
    t = 0
    prob_total = 0
    for x in range(400, 601):
        relative_x = x - 400
        value = getState(200, energy, relative_x)
        prob_total += (value * complex.conjugate(value)).real
        field[t, x, psi_index] = value
        field[t, x, boundary_index] = complex(1, 0)
    print(f'Probability: {prob_total}')

    t = 999
    for x in range(400, 601):
        relative_x = x - 400
        value = getState(200, energy, relative_x)
        field[t, x, psi_index] = value
        field[t, x, boundary_index] = complex(1, 0)


def getBestState2(field, x, t):
    val_tp1 = field[t + 1, x, psi_index]
    val_tn1 = field[t - 1, x, psi_index]
    val_xp1 = field[t, x + 1, psi_index]
    val_xn1 = field[t, x - 1, psi_index]
    pot = field[t, x, potential_index]

    best_psi = (0.5 * (val_tp1 - val_tn1) * complex(0, 1) + val_xp1 + val_xn1)/(2 + pot)

    return best_psi, 0.0


def normalize(field, t, particles):
    x_extent = field.shape[1]
    sum_prob = 0
    for x in range(x_extent):
        if field[t, x, boundary_index] != complex(1, 0):
            sum_prob += (field[t, x, psi_index] * complex.conjugate(field[t, x, psi_index])).real
    for x in range(x_extent):
        if field[t, x, boundary_index] != complex(1, 0):
            field[t, x, psi_index] = particles * field[t, x, psi_index] / sum_prob


def fit2(iterations, field, particles):
    t_extent = field.shape[0]
    x_extent = field.shape[1]

    for idx in range(iterations):
        print(f'Iteration: {idx}')

        for t in range(1, t_extent-1):
            r = random.randint(0, 1)
            if r == 0:
                for x in range(1, x_extent - 1):
                    if field[t, x, boundary_index] != complex(1, 0):
                        min_psi, error = getBestState2(field, x, t)
                        field[t, x, psi_index] = min_psi
            else:
                for x in range(x_extent - 1, -1, -1):
                    if field[t, x, boundary_index] != complex(1, 0):
                        min_psi, error = getBestState2(field, x, t)
                        field[t, x, psi_index] = min_psi
            normalize(field, t, particles)

        for t in range(t_extent-1, -1, -1):
            r = random.randint(0, 1)
            if r == 0:
                for x in range(1, x_extent - 1):
                    if field[t, x, boundary_index] != complex(1, 0):
                        min_psi, error = getBestState2(field, x, t)
                        field[t, x, psi_index] = min_psi
            else:
                for x in range(x_extent - 1, -1, -1):
                    if field[t, x, boundary_index] != complex(1, 0):
                        min_psi, error = getBestState2(field, x, t)
                        field[t, x, psi_index] = min_psi
            normalize(field, t, particles)


def plotSolution(field):
    t_extent = field.shape[0]
    x_extent = field.shape[1]

    # img = np.zeros((t_extent,x_extent))
    #
    # for x in range(0, x_extent):
    #     for t in range(0, t_extent):
    #         prob = (field[t, x, psi_index] * complex.conjugate(field[t, x, psi_index])).real
    #         img[t, x] = prob
    #
    # # plt.imshow(img, cmap='gray')
    # plt.imshow(img)
    # plt.title('Probability')
    # plt.show()

    t_range = [0, 1, 2, 3, 500, 997, 998, 999]

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


def main():
    particles = 1
    energy = 2
    field = createField(1000, 1000)
    setWellPotential(field, particles)
    setWellBoundary(field, energy)
    iterations = 40
    fit2(iterations, field, particles)
    plotSolution(field)


main()
