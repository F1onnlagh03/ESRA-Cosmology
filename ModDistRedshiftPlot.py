import matplotlib.pyplot as plt
import numpy as np
import math as m

# Create a list such that each element is a line from the Data.txt file
data = [line.rstrip() for line in open('Data.txt')]

# Delete first 5 terms from list
del data[0:5]

data_array = []
for i in range(0, len(data)-1):
    data_array.append(data[i].split())

# Data is organised as: Supernova Name, Redshift, Distance Modulus, Distance Modulus Error,
# probability that the supernova was hosted by a low-mass galaxy

distance_modulus = []
distance_modulus_error = []
redshift = []
for i in range(0, len(data)-1): # separate data into individual lists
    distance_modulus.append(float(data_array[i][2]))
    redshift.append(float(data_array[i][1]))
    distance_modulus_error.append(float(data_array[i][3]))

# find mins and maxs
min_d = float(min(distance_modulus))
max_d = float(max(distance_modulus))
min_r = float(min(redshift))
max_r = float(max(redshift))

plt.figure(figsize=(12,8))
plt.errorbar(distance_modulus, redshift,
             yerr=distance_modulus_error,
             elinewidth=1, capsize=1, fmt='o',
             alpha=0.5, ms=0.7)
plt.xlabel('Distance modulus')
plt.ylabel('redshift')
plt.xlim(min_d - 0.5, max_d + 0.5)
plt.ylim(min_r - 0.7, max_r + 0.7)
plt.show()
