import matplotlib.pyplot as plt
import numpy as np
import math as m
import astropy.constants as const
import scipy.stats as stats
import scipy.integrate as integrate

# Create a list such that each element is a line from the Data.txt file
data = [line.rstrip() for line in open('Data.txt')]
open('Data.txt').close()

# Delete first 5 terms from list
del data[0:5]

# Data is organised as: Supernova Name, Redshift, Distance Modulus, Distance Modulus Error,
# probability that the supernova was hosted by a low-mass galaxy

data_array = []
for i in range(0, len(data)-1):
    data_array.append(data[i].split())

distance_modulus = []
distance_modulus_error = []
redshift = []
for i in range(0, len(data_array)-1): # separate data into individual lists
    distance_modulus.append(float(data_array[i][2]))
    redshift.append(float(data_array[i][1]))
    distance_modulus_error.append(float(data_array[i][3]))

redshift = np.array(redshift)

# Assuming a flat universe

# Model A
omega_m = 1
omega_lambda = 0

# create function
def E(redshift):
    return m.sqrt(omega_m * (1+redshift)**3 + omega_lambda)

print('here')
I = integrate.quad(E, 0, redshift[-1])
d_L = (1 + redshift) * I

print(d_L)
