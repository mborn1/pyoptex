from pyoptex.doe.utils.comp import outer_integral
import numpy as np

n = 10000000
samples = np.stack((
    np.ones(n),
    np.random.rand(n) * 2 - 1,
    np.random.rand(n) * 2 - 1,
))
samples = np.stack((
    samples[0], samples[1], samples[2], samples[1] * samples[2], samples[1]**2, samples[2]**2
))
print(samples.shape)

print(outer_integral(samples.T))
