import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("timing_results.csv", delimiter=",", skiprows=1)
num_procs = data[:, 0].astype(int)
times = data[:, 1]

t1 = times[num_procs == 1][0]
speedup = t1 / times

plt.figure()
plt.plot(num_procs, speedup, marker='o', label='Measured speedup')
plt.plot(num_procs, num_procs, linestyle='--', label='Ideal speedup')
plt.xlabel("Number of processes")
plt.ylabel("Speedup")
plt.title("Mandelbrot parallel speedup")
plt.legend()
plt.xticks(num_procs)
plt.tight_layout()
plt.savefig("speedup.png")
