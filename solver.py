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
    # field = np.empty_like([[[1+1j,float(1.0),int(1)]]],shape=(t_extent,x_extent,3))
    field = np.zeros((t_extent, x_extent, 3), dtype=complex)

    # for t in range(t_extent):
    #     for x in range(x_extent):
    #         field[t,x,complex_index] = complex(0.0, 0.0)
    return field


def setWellPotential(field):
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

    # Set initial values

    for x in range(400, 600):
        for t in range(t_extent):
            r = random.uniform(0, 0.01)
            i = random.uniform(0, 0.01)
            field[t, x, psi_index] = complex(0.1, 0.1)


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
        value = getState(200, 1, relative_x)
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

    best_psi = (0.5*(val_tp1 - val_tn1) * complex(0,1) - val_xp1 + val_xn1)/(2 + pot)

    return best_psi, 0.0


# def getBestState(field, x, t, learning_rate):
#
#     psi_collection = []
#     unit_vector = complex(1, 0) * learning_rate
#     theta_list = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
#
#     psi = field[t, x, psi_index]
#     for theta in theta_list:
#         psi_new = unit_vector * (theta / 360) * complex(0, 1) + psi
#         psi_collection.append((psi_new, getError(field, x, t, psi_new)))
#
#     # psi = field[t, x, psi_index]
#     # psi_collection.append((psi, getError(field, x, t, psi)))
#     #
#     # psi_1 = psi + learning_rate
#     # psi_collection.append((psi_1, getError(field, x, t, psi_1)))
#     #
#     # psi_2 = psi - learning_rate
#     # psi_collection.append((psi_2, getError(field, x, t, psi_2)))
#     #
#     # psi_3 = psi + learning_rate * complex(0, 1)
#     # psi_collection.append((psi_3, getError(field, x, t, psi_3)))
#     #
#     # psi_4 = psi - learning_rate * complex(0, 1)
#     # psi_collection.append((psi_4, getError(field, x, t, psi_4)))
#
#     idx = 0
#     for item in psi_collection:
#         if idx == 0:
#             min_error = item[1]
#             best_psi = item[0]
#         else:
#             if item[1] < min_error:
#                 min_error = item[1]
#                 best_psi = item[0]
#         idx += 1
#
#     return best_psi, min_error.real
#
#
# def getError(field, x, t, psi):
#
#     val_tp1 = field[t + 1, x, psi_index]
#     val_tn1 = field[t - 1, x, psi_index]
#     val_xp1 = field[t, x + 1, psi_index]
#     val_xn1 = field[t, x - 1, psi_index]
#     pot = field[t, x, potential_index]
#
#     t_term = 0.5 * (val_tp1 - val_tn1) * complex(0, 1)
#     x_term = val_xp1 - 2 * psi + val_xn1
#     v_term = pot * psi
#     error = t_term + x_term - v_term
#
#     # Pick slope from front and from behind evenly
#     # r = random.randint(0,1)
#     # if r == 0:
#     #     error = (field[t + 1, x, psi_index] - psi) * complex(0, 1) \
#     #             + (field[t, x + 1, psi_index] - 2 * psi + field[t, x - 1, psi_index]) \
#     #             - field[t, x, potential_index] * psi
#     # if r == 1:
#     #     error = (psi - field[t - 1, x, psi_index]) * complex(0, 1) \
#     #             + (field[t, x + 1, psi_index] - 2 * psi + field[t, x - 1, psi_index]) \
#     #             - field[t, x, potential_index] * psi
#
#     error_squared = error * complex.conjugate(error)
#     return error_squared
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

        #
        # for x in range(1, x_extent-1):
        #     for t in range(1, t_extent-1):
        #         if field[t, x, boundary_index] != complex(1, 0):
        #             min_psi, error = getBestState2(field, x, t)
        #             field[t, x, psi_index] = min_psi
        # for x in range(x_extent-1, -1, -1):
        #     for t in range(t_extent-1, -1, -1):
        #         if field[t, x, boundary_index] != complex(1, 0):
        #             min_psi, error = getBestState2(field, x, t)
        #             field[t, x, psi_index] = min_psi



# def fit(iterations, field, learning_rate):
#     t_extent = field.shape[0]
#     x_extent = field.shape[1]
#     effective_lr = learning_rate
#
#     error_list = []
#     first_value = True
#     for idx in range(iterations):
#         iteration_error = 0
#         for x in range(1, x_extent-1):
#             for t in range(1, t_extent-1):
#                 if field[t, x, boundary_index] != complex(1, 0):
#                     min_psi, error = getBestState2(field, x, t, effective_lr)
#                     field[t, x, psi_index] = min_psi
#                     iteration_error += error
#         if first_value:
#             first_error = iteration_error
#             first_value = False
#         else:
#             effective_lr = learning_rate * (iteration_error/first_error)
#         error_list.append(iteration_error)
#         print(f'Iteration: {idx}   Error: {iteration_error}')
#     return error_list


def plotSolution(field, error_list):
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
        plt.plot(real_list)
        plt.plot(im_list)
        plt.legend(['prob', 'real', 'imag'])
        plt.title(f'T={t}')
        plt.show()

    # plt.plot(error_list)
    # plt.title('Errors')
    # plt.show()


def main():
    particles = 1
    energy = 2
    field = createField(1000, 1000)
    setWellPotential(field)
    setWellBoundary(field, energy)
    iterations = 10
    learning_rate = 0.0001
    error_list = []
    # error_list = fit(iterations, field, learning_rate)
    fit2(iterations, field, particles)
    plotSolution(field, error_list)


main()
