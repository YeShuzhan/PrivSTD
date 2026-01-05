import numpy
import numpy as np
import math


def UG(data: numpy.ndarray, c=10, eps=1.0, sigma=1.0):
    data_noisy = np.zeros_like(data)
    N = data.sum()
    dim = data.shape[0]

    num_grids = int(math.sqrt(N * eps / c))
    num_grids = max(1, num_grids)
    num_grids = min(num_grids, dim)

    grid_size = round(dim // num_grids)

    grids = []
    x0, y0, x_size, y_size = 0, 0, 0, 0
    for i in range(0, dim, grid_size):
        for j in range(0, dim, grid_size):
            x0, y0, x_size, y_size = i, j, min(i + grid_size, dim), min(j + grid_size, dim)
            grids.append((x0, y0, x_size, y_size))

    # last grid
    if x_size < dim or y_size < dim:
        x0, y0, x_size, y_size = x_size, y_size, dim, dim
        grids.append((x0, y0, x_size, y_size))

    for grid in grids:
        x0, y0, x_1, y_1 = grid
        count = data[x0:x_1, y0:y_1].sum()
        noisy_count = np.random.normal(count, sigma)
        average = noisy_count / (x_1 - x0) / (y_1 - y0)
        data_noisy[x0:x_1, y0:y_1] = average

    return data_noisy


def AG(data: numpy.ndarray, c1=10, c2=5, alpha=0.5, eps=1.0, delta=0.0):
    data_noisy = np.zeros_like(data)
    N = data.sum()
    dim = data.shape[0]

    num_grids_level1 = max(10, int(1 / 4 * math.sqrt(N * eps / c1)))
    num_grids_level1 = min(num_grids_level1, dim)
    num_grids_level1 = max(1, num_grids_level1)
    grid_size_level1 = round(dim / num_grids_level1)

    grids = []
    for i in range(0, dim, grid_size_level1):
        for j in range(0, dim, grid_size_level1):
            x0, y0, x_size, y_size = i, j, min(i + grid_size_level1, dim), min(j + grid_size_level1, dim)
            grids.append((x0, y0, x_size, y_size))

            # add noise to first level grids
            count = data[x0:x_size, y0:y_size].sum()
            sigma = math.sqrt(2 * math.log(1.25 / (alpha * delta))) / (alpha * eps)
            noisy_count = np.random.normal(count, sigma)
            average = noisy_count / (x_size - x0) / (y_size - y0)
            data_noisy[x0:x_size, y0:y_size] = average

            # second-level grids
            if count > 1:
                num_grids_level2 = math.ceil(math.sqrt(count * (1 - alpha) * eps / c2))
                num_grids_level2 = min(num_grids_level2, x_size - x0)
                num_grids_level2 = max(1, num_grids_level2)
                grid_size_level2 = round((x_size - x0) / num_grids_level2)
            else:
                grid_size_level2 = x_size - x0

            sum_count_noisy_local = 0
            local_grids = []
            local_noises = []
            for p in range(0, x_size - x0, grid_size_level2):
                for q in range(0, y_size - y0, grid_size_level2):
                    x0_, y0_, x_size_, y_size_ = x0 + p, y0 + q, min(x0 + p + grid_size_level2, x_size), min(
                        y0 + q + grid_size_level2, y_size)
                    local_grids.append((x0_, y0_, x_size_, y_size_))
                    count_local = data_noisy[x0_:x_size_, y0_:y_size_].sum()
                    sigma = math.sqrt(2 * math.log(1.25 / ((1-alpha)*delta))) / ((1 - alpha) * eps)
                    noisy_count_local = np.random.normal(count_local, sigma)
                    local_noises.append(noisy_count_local)
                    sum_count_noisy_local += noisy_count_local

            # consistent post-processing
            m2 = grid_size_level2
            a = (alpha ** 2) * (m2 ** 2) / ((1 - alpha) ** 2 + (alpha ** 2) * (m2 ** 2))
            noisy_count = a * noisy_count + (1 - a) * sum_count_noisy_local

            diff = (noisy_count - sum_count_noisy_local) / len(local_grids)

            for local_grid, local_noise in zip(local_grids, local_noises):
                x0_, y0_, x_size_, y_size_ = local_grid
                data_noisy[x0_:x_size_, y0_:y_size_] = (local_noise + diff) / (x_size_ - x0_) / (y_size_ - y0_)
    return data_noisy
