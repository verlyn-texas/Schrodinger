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

    for x in range(0, 200):
        for t in range(t_extent):
            field[t, x, potential_index] = complex(1, 0)
            field[t, x, boundary_index] = complex(1, 0)

    for x in range(401, 600):
        for t in range(t_extent):
            field[t, x, potential_index] = complex(1, 0)
            field[t, x, boundary_index] = complex(1, 0)

    for x in range(801, 1000):
        for t in range(t_extent):
            field[t, x, potential_index] = complex(1, 0)
            field[t, x, boundary_index] = complex(1, 0)

    # Set initial values in well
    for t in range(t_extent):
        for x in range(200, 401):
            field[t, x, psi_index] = complex(0.1, 0.1)
        for x in range(600, 801):
            field[t, x, psi_index] = complex(0.1, 0.1)
        normalize(field, t, particles)

def setTrenchPotential(field, particles):
    x_start = 401
    x_end = 599
    t_start = 400
    t_end = 600
    potential = 0.0

    for t in range(t_start,t_end+1):
        for x in range(x_start,x_end+1):
            field[t, x, potential_index] = complex(potential, 0)
            field[t, x, boundary_index] = complex(potential, 0)
            field[t, x, psi_index] = complex(0.1, 0.1)
        normalize(field, t, particles)

def getState(width, n, x):
    real = math.sqrt(2 / width) * math.sin(n * math.pi * x / width)
    imaginary = 0
    value = complex(real, imaginary)
    return value

def setWellBoundary(field, low_energy, high_energy):
    # Set boundary conditions
    t = 0
    prob_total = 0
    for x in range(200, 401):
        relative_x = x - 200
        value = getState(200, low_energy, relative_x)
        prob_total += (value * complex.conjugate(value)).real
        field[t, x, psi_index] = value
        field[t, x, boundary_index] = complex(1, 0)
    print(f'Probability Left: {prob_total} at t={t}')
    prob_total = 0
    for x in range(600, 801):
        relative_x = x - 600
        value = getState(200, high_energy, relative_x)
        prob_total += (value * complex.conjugate(value)).real
        field[t, x, psi_index] = value
        field[t, x, boundary_index] = complex(1, 0)
    print(f'Probability Right: {prob_total} at t={t}')

    t = 999
    prob_total = 0
    for x in range(200, 401):
        relative_x = x - 200
        value = getState(200, high_energy, relative_x)
        prob_total += (value * complex.conjugate(value)).real
        field[t, x, psi_index] = value
        field[t, x, boundary_index] = complex(1, 0)
    print(f'Probability Left: {prob_total} at t={t}')
    prob_total = 0
    for x in range(600, 801):
        relative_x = x - 600
        value = getState(200, low_energy, relative_x)
        prob_total += (value * complex.conjugate(value)).real
        field[t, x, psi_index] = value
        field[t, x, boundary_index] = complex(1, 0)
    print(f'Probability Right: {prob_total} at t={t}')


def getBestState2(field, x, t, h):
    val_tp1 = field[t + 1, x, psi_index]
    val_tn1 = field[t - 1, x, psi_index]
    val_xp1 = field[t, x + 1, psi_index]
    val_xn1 = field[t, x - 1, psi_index]
    pot = field[t, x, potential_index]

    best_psi = (h * 0.5 * (val_tp1 - val_tn1) * complex(0, 1) + h * h * (val_xp1 + val_xn1))/(2 * h * h + pot)

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


def fit2(iterations, field, particles, h):
    t_extent = field.shape[0]
    x_extent = field.shape[1]

    for idx in range(iterations):
        print(f'Iteration: {idx}')
        # Forward In Time
        raster_right = True
        for t in range(1, t_extent - 1):
            if raster_right:
                for x in range(1, x_extent - 1):
                    if field[t, x, boundary_index] != complex(1, 0):
                        min_psi, error = getBestState2(field, x, t, h)
                        field[t, x, psi_index] = min_psi
                raster_right = not raster_right
            else:
                for x in range(x_extent - 1, -1, -1):
                    if field[t, x, boundary_index] != complex(1, 0):
                        min_psi, error = getBestState2(field, x, t, h)
                        field[t, x, psi_index] = min_psi
                raster_right = not raster_right

            normalize(field, t, particles)
        # Backward in Time
        raster_right = True
        for t in range(t_extent-1, -1, -1):
            if raster_right:
                for x in range(1, x_extent - 1):
                    if field[t, x, boundary_index] != complex(1, 0):
                        min_psi, error = getBestState2(field, x, t, h)
                        field[t, x, psi_index] = min_psi
                raster_right = not raster_right
            else:
                for x in range(x_extent - 1, -1, -1):
                    if field[t, x, boundary_index] != complex(1, 0):
                        min_psi, error = getBestState2(field, x, t, h)
                        field[t, x, psi_index] = min_psi
                raster_right = not raster_right

            normalize(field, t, particles)


def plotSolution(field):
    t_extent = field.shape[0]
    x_extent = field.shape[1]

    img = np.zeros((t_extent,x_extent))

    for x in range(0, x_extent):
        for t in range(0, t_extent):
            # prob = (field[t, x, psi_index] * complex.conjugate(field[t, x, psi_index])).real
            prob = (field[t, x, potential_index]).real
            img[t, x] = prob

    plt.imshow(img, cmap='gray', vmin=0.0, vmax=0.02)
    # plt.imshow(img, vmin=0.0, vmax=0.02)
    # plt.title('Probability')
    plt.title('Potential')
    plt.show()

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
        plt.plot(real_list)
        plt.plot(im_list)
        plt.legend(['prob', 'real', 'imag'])
        plt.title(f'T={t}')
        plt.show()


def main():
    particles = 2
    low_energy = 1
    high_energy = 2
    h = 0.1
    field = createField(1000, 1000)
    setWellPotential(field, particles)
    setTrenchPotential(field, particles)
    setWellBoundary(field, low_energy, high_energy)
    iterations = 5
    fit2(iterations, field, particles, h)
    plotSolution(field)


main()