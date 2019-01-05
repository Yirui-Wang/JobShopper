from random import sample, randint

import numpy as np

if __name__ == '__main__':
    n_machines = 20
    n_jobs = 50

    machine_matrix = [sample(range(n_machines), n_machines) for _ in range(n_jobs)]
    time_matrix = [[randint(0, 100) for _ in range(n_machines)] for _ in range(n_jobs)]

    np.savetxt('data/machine.csv', machine_matrix, fmt='%d', delimiter=',')
    np.savetxt('data/time.csv', time_matrix, fmt='%d', delimiter=',')
