"""
Generates fake continuous visibility data for
debugging purposes.

Randomly chooses (x, y, 0) values and assigns
a visibility error score of (euclidean distance)
to the obstacle
"""

import numpy as np
import math
import csv

NDATA = 10000

# x values [0, 50]
# y values [-50, 50]

def generate_continuous(n):
    data = []

    for i in range(NDATA):
        x_cur = np.random.randint(n)
        y_cur = np.random.randint(-n, n)
        z_cur = 0

        error = math.sqrt(x_cur ** 2 + y_cur ** 2)
        data.append([x_cur, y_cur, z_cur, error])

    return data

def generate_discrete(n, boundary_dist=45):
    data = []

    for i in range(NDATA):
        x_cur = np.random.randint(n)
        y_cur = np.random.randint(-n, n)
        z_cur = 0

        dist = math.sqrt(x_cur ** 2 + y_cur ** 2)

        if (dist < boundary_dist):
            error = 0.0001
        else:
            error = 1.0

        data.append([x_cur, y_cur, z_cur, error])

    return data

def write_csv(data, filename):
    with open(filename, mode='w') as datafile:
        vis_writer = csv.writer(datafile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        vis_writer.writerow(["X", "Y", "Z", "ERROR"])
        for i in range(len(data)):
            vis_writer.writerow(data[i])


if __name__ == "__main__":
    #data = generate_continuous(50)
    data = generate_discrete(50)
    #print(data)
    #write_csv(data, "fakecont.csv")
    write_csv(data, "fakedis.csv")
