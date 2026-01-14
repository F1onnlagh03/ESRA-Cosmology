
import matplotlib.pyplot as plt
import numpy as np
import math as m
import astropy.constants as const

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

# mu = 5log(D) + 25
# D = 10^[(mu-25)/5] for luminosity distance
# D_error = ln(10)*(10^((distance_modulus[i]-25)/5)/5)

D = []
D_error = []
# Calculate the distance luminosty
for i in range(0, len(distance_modulus)):
    D.append(10**((distance_modulus[i]-25)/5))
    D_error.append(m.log(10)*(10**((distance_modulus[i]-25)/5)/5))

# Calculate recession velocity
c = const.c.value
v = []
for i in range(0, len(redshift)):
    v.append(c*redshift[i])

# plot recession velocity against luminosity distance
plt.figure(figsize=(12,8))
plt.errorbar(v, D,
             yerr=D_error,
             elinewidth=1, capsize=1, fmt='o',
             alpha=0.5, ms=0.7)
plt.xlabel('recession velocity (m/s)')
plt.ylabel('luminosity distance')

# plot redshift against luminosity distance
plt.figure(figsize=(12,8))
plt.errorbar(redshift, D,
             yerr=D_error,
             elinewidth=1, capsize=1, fmt='o',
             alpha=0.5, ms=0.7)
plt.xlabel('redshift')
plt.ylabel('luminosity distance')


# plot redshift against distance modulus
plt.figure(figsize=(12,8))
plt.errorbar(redshift, distance_modulus,
             yerr=distance_modulus_error,
             elinewidth=1, capsize=1, fmt='o',
             alpha=0.5, ms=0.7)
plt.ylabel('Distance modulus')
plt.xlabel('Redshift')
plt.ylim(min_d - 0.5, max_d + 0.5)
plt.xlim(min_r - 0.01, max_r + 0.01)
plt.show()
