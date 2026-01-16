import matplotlib.pyplot as plt
import numpy as np
import math as m
import astropy.constants as const
import scipy.stats as stats

# Create a list such that each element is a line from the Data.txt file
data = [line.rstrip() for line in open('Data.txt')]

# Delete first 5 terms from list
del data[0:5]

# Data is organised as: Supernova Name, Redshift, Distance Modulus, Distance Modulus Error,
# probability that the supernova was hosted by a low-mass galaxy

data_array = []
for i in range(0, len(data)-1):
    data_array.append(data[i].split())

# filter out objects with z >= 0.025
close_data_array = []
for i in range(0, len(data_array)):
    if float(data_array[i][1]) < 0.025:
        close_data_array.append(data_array[i])
    else:
        continue

low_distance_modulus = []
low_distance_modulus_error = []
low_redshift = []
for i in range(0, len(close_data_array)-1): # separate data into individual lists
    low_distance_modulus.append(float(close_data_array[i][2]))
    low_redshift.append(float(close_data_array[i][1]))
    low_distance_modulus_error.append(float(close_data_array[i][3]))

distance_modulus = []
distance_modulus_error = []
redshift = []
for i in range(0, len(data_array)-1): # separate data into individual lists
    distance_modulus.append(float(data_array[i][2]))
    redshift.append(float(data_array[i][1]))
    distance_modulus_error.append(float(data_array[i][3]))

#### WORKING ####
# mu = 5log(D) - 5
# D = 10^[mu/5 + 1] for luminosity distance in pc
# D_error = 0.2 * ln(10) * 10^(0.2 * mu + 1)
#################

low_D = []
low_D_error = []
# Calculate the Distance luminosity
for i in range(0, len(low_distance_modulus)):
    low_D.append(10**(low_distance_modulus[i]/5 + 1)/1e6)
    low_D_error.append(0.2*m.log(10)*10**(0.2*low_distance_modulus_error[i]+1)/1e6)

D = []
D_error = []
# Calculate the Distance luminosity
for i in range(0, len(distance_modulus)):
    D.append(10**(distance_modulus[i]/5 + 1)/1e6)
    D_error.append(0.2*m.log(10)*10**(0.2*distance_modulus_error[i]+1)/1e6)

# Calculate the non-relativistic recession velocity
c = const.c.to('km/s').value # convert to km/s
low_v = []
for i in range(0, len(low_redshift)): # low z regime
    low_v.append(c*low_redshift[i])
v = []
for i in range(0, len(redshift)): # considering all z values
    v.append(c*redshift[i])

# Calculate the relativistic recession velocity
low_rel_v = []
for i in range(0, len(low_redshift)):
    low_rel_v.append(c*((low_redshift[i]+1)**2-1)/((low_redshift[i]+1)**2+1))
rel_v = []
for i in range(0, len(redshift)):
    rel_v.append(c*((redshift[i]+1)**2-1)/((redshift[i]+1)**2+1))

# use linear regression to find best fit line of data
low_res = stats.linregress(low_v, low_D)
res = stats.linregress(v, D)

low_rel_res = stats.linregress(low_rel_v, low_D)
rel_res = stats.linregress(rel_v, D)


# vectorise the list
low_v = np.array(low_v)
v = np.array(v)
low_rel_v = np.array(low_rel_v)
rel_v = np.array(rel_v)

# Control what set of data to display
# graph_no = 0 (only Hubbles constant), 1 (hubbles constant and 4 plots it is computed from),
# 2 (redshift against luminosity distance), 3 (redshift against distance modulus)

graph_no = 1

if graph_no == 0:
    print(f'Non-relativistic Hubbles Constant (z<0.025) = {1/low_res.slope} ± {low_res.stderr}')
    print(f'Non-relativistic Hubbles Constant = {1/res.slope} ± {res.stderr}')
    print(f'Relativistic Hubbles Constant (z<0.025) = {1/low_rel_res.slope} ± {low_rel_res.stderr}')
    print(f'Relaticistic Hubbles Constant = {1/rel_res.slope} ± {rel_res.stderr}')

if graph_no == 1:
    # plot recession velocity against luminosity distance
    fig, ax = plt.subplot_mosaic([['low_redshift', 'norm_redshift'],
                                  ['rel_low_r', 'rel_norm_r']],
                                 figsize=[10,6])
    # Non-relativistic Case
    ax['low_redshift'].errorbar(low_D, low_v,
                 xerr=low_D_error,
                 elinewidth=1, capsize=1, fmt='o', c='b',
                 alpha=0.5, ms=0.7, label='Non-relativistic')
    ax['low_redshift'].plot(low_res.intercept + low_res.slope*low_v, low_v,
                            c='r', label='fitted non-relativistic')
    ax['norm_redshift'].errorbar(D, v,
                 xerr=D_error,
                 elinewidth=1, capsize=1, fmt='o', c='g',
                 alpha=0.5, ms=0.7, label='Non-relativistic')
    ax['norm_redshift'].plot(res.intercept + res.slope*v, v, c='y', label='fitted data')

    # Relativistic Case
    ax['rel_low_r'].errorbar(low_D, low_rel_v,
                                xerr=low_D_error,
                                elinewidth=1, capsize=1, fmt='o', c='b',
                                alpha=0.5, ms=0.7, label='Relativistic')
    ax['rel_low_r'].plot(low_rel_res.intercept + low_rel_res.slope * low_rel_v, low_rel_v,
                            c='r', label='fitted data')
    ax['rel_norm_r'].errorbar(D, rel_v,
                                 xerr=D_error,
                                 elinewidth=1, capsize=1, fmt='o', c='g',
                                 alpha=0.5, ms=0.7, label='Relativistic')
    ax['rel_norm_r'].plot(rel_res.intercept + rel_res.slope * rel_v, rel_v,
                          c='y', label='fitted data')

    ax['rel_low_r'].set_ylabel('recession velocity (km/s)')
    ax['low_redshift'].set_ylabel('recession velocity (km/s)')
    ax['rel_low_r'].set_xlabel('luminosity distance (Mpc)')
    ax['rel_norm_r'].set_xlabel('luminosity distance (Mpc)')

    print(f'Non-relativistic Hubbles Constant (z<0.025) = {1/low_res.slope} ± {low_res.stderr} km/s/Mpc')
    print(f'Non-relativistic Hubbles Constant = {1/res.slope} ± {res.stderr} km/s/Mpc')
    print(f'Relativistic Hubbles Constant (z<0.025) = {1/low_rel_res.slope} ± {low_rel_res.stderr} km/s/Mpc')
    print(f'Relaticistic Hubbles Constant = {1/rel_res.slope} ± {rel_res.stderr} km/s/Mpc')
    plt.show()

if graph_no == 2:
# plot redshift against luminosity distance
    plt.figure(figsize=(12,8))
    plt.errorbar(low_redshift, low_D,
                 yerr=low_D_error,
                 elinewidth=1, capsize=1, fmt='o',
                 alpha=0.5, ms=0.7)
    plt.xlabel('redshift')
    plt.ylabel('luminosity distance')
    plt.show()

if graph_no == 3:
# plot redshift against distance modulus
    plt.figure(figsize=(12,8))
    plt.errorbar(low_redshift, low_distance_modulus,
                 yerr=low_distance_modulus_error,
                 elinewidth=1, capsize=1, fmt='o',
                 alpha=0.5, ms=0.7)
    plt.ylabel('Distance modulus')
    plt.xlabel('Redshift')
    plt.show()
