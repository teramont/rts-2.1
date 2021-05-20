import numpy as np
import matplotlib.pyplot as plt
from time import time

HARMONICS_COUNT = 6
MAX_FREQUENCY = 1700
DISCRETE_TIMES_COUNT = 1024

def rand_sig(harmonics_count, max_freq, discr_times_count):
	sig = np.zeros(discr_times_count)
	freq_start = max_freq / harmonics_count
	for harmonic_index in range(harmonics_count):
		amplitude = np.random.uniform(0.0, 1000.0)
		phase = np.random.uniform(-np.pi / 2, np.pi / 2)
		freq = freq_start * (harmonic_index + 1)
		for time in range(discr_times_count):
			sig[time] += amplitude * np.sin(freq * time + phase)
	return sig

def discrete_fourier_transform(sig):
	res = np.zeros(len(sig))
	for p in range(len(sig)):
		sum = 0
		for k in range(len(sig)):
			angle = 2 * np.pi * p * k / len(sig)
			turn_coef = complex(np.cos(angle), -np.sin(angle))
			sum += sig[k] * turn_coef
		res[p] = abs(sum)
	return res


#Additional task

dft_durations = np.zeros(10)
numpy_durations = np.zeros(10)

for i in range(1, 10):
	N = i * 64
	sig = rand_sig(HARMONICS_COUNT, MAX_FREQUENCY, N)
	before = time()
	discrete_fourier_transform(sig)
	after = time()
	dft_durations[i] = after - before
	before = time()
	np.fft.fft(sig)
	after = time()
	numpy_durations[i] = after - before

plt.plot(range(10), dft_durations, label = "custom")
plt.plot(range(10), numpy_durations, label = "numpy")
plt.legend()
plt.xlabel("N")
plt.ylabel("DFT value")
plt.show()