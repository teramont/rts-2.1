import numpy as np
import matplotlib.pyplot as plt

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

sig = rand_sig(HARMONICS_COUNT, MAX_FREQUENCY, DISCRETE_TIMES_COUNT)

DFT = discrete_fourier_transform(sig)

plt.plot(range(DISCRETE_TIMES_COUNT), DFT)
plt.xlabel("p value")
plt.ylabel("DFT value")
plt.show()