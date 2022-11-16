import numpy as np
alphas = np.random.random(6)
# print(alphas)
mbr_degrees = np.zeros([6, 3])
for i in range(6):
    mbr_degrees[i][np.random.randint(3)] = 1.
# print(mbr_degrees)
ksi = 2.*np.random.random(6) - 1.
# print(ksi)
beta = 2.*np.random.random([6, 3]) - 1.
# print(beta)
gradIdxs = [0, 0, 0]
prevGrad = np.zeros(54)
# print(prevGrad)
# print(prevGrad.shape)
desiredBBAs = [ np.zeros(3) for i in range(150) ]

freeDims = list(range(4))
numPoints = np.product([91, 51])
print(np.random.random(2))

