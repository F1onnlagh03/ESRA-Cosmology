import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
from scipy import ndimage

class Hubbles():

    def __init__(self, SN, z, mu, mu_err):

        self.SN = SN
        self.z = z
        self.mu = mu
        self.mu_err = mu_err

    def organise_data(self):

        """
        Organises the data by size for plotting and use

        """

        sort_index = np.argsort(self.z)
        self.z = self.z[sort_index]
        self.mu = self.mu[sort_index]
        self.mu_err = self.mu_err[sort_index]

    def plot_hubble(self):

        """
        Plots the redshift against the distance modulus
        the hubble constant can be derived from this information

        """
        plt.errorbar(self.z, self.mu, yerr=self.mu_err, fmt='o')
        plt.xlabel('Redshift, z')
        plt.ylabel('Distance Modulus')
        plt.show()

    def begin_hubble(self):

        """
        Initiate plotting the hubble diagram

        """

        self.organise_data()
        self.plot_hubble()

    def begin_other(self):


        """
        Initiate finding the cosmological constants

        """
        self.organise_data()

        H0, OmegaM, OmegaLambda = self.create_consts()

        DL_values = self.dist_lum2(H0, OmegaM, OmegaLambda)

        calced_dist_mod = self.calced_dist_mod(DL_values)

        #print(calced_dist_mod)
        print("Calculated Distance Modulus")
        chi2_grid = self.get_chi_squared(calced_dist_mod)
        #print(chi2_grid)
        #self.plot_chi2(chi2_grid)

        H0_semifinal = []
        OmegaM_semifinal = []
        OmegaLambda_semifinal = []
        #for i in range(0, 100):
        #    print(f"Iteration {i}")
        i0, j0, k0, path = self.find_minima(chi2_grid)
        #    H0_semifinal.append(H0[i0])
        #    OmegaM_semifinal.append(OmegaM[j0])
        #    OmegaLambda_semifinal.append(OmegaLambda[k0])

        #min_pos = self.find_minima2(chi2_grid)
        #print(min_pos)
        #i0, j0, k0 = min_pos

        print(f"H_0 = {H0[i0]}")
        print(f"OmegaM = {OmegaM[j0]}")
        print(f"OmegaLambda = {OmegaLambda[k0]}")

        self.plot_triangle(chi2_grid, H0, OmegaM, OmegaLambda)

        #H0_mean = np.mean(H0_semifinal)
        #OmegaM_mean = np.mean(OmegaM_semifinal)
        #OmegaLambda_mean = np.mean(OmegaLambda_semifinal)

        #print(rf"$H_0$ = {H0_mean}")
        #print(rf"$\Omega_M$ = {OmegaM_mean}")
        #print(rf"$\Omega_Lambda$ = {OmegaLambda_mean}")

    def create_consts(self):

        """
        Create an array of potential values of cosmological constants

        """
        H0 = np.arange(50, 110, 1)
        OmegaM = np.arange(0.2, 1.0, 0.02)
        OmegaLambda = 1 - OmegaM
        #OmegaLambda = np.arange(-2, 3, 0.02)

        return  H0, OmegaM, OmegaLambda

    def dist_lum2(self, H0, OmegaM, OmegaLambda):
        """
        Calculate distance luminosity following Riess et al. 1998 equation (2)
        """
        c = 3e5  # km/s
        H0_grid, OmegaM_grid, OmegaLambda_grid = np.meshgrid(H0, OmegaM, OmegaLambda, indexing='ij')
        OmegaK_grid = 1 - OmegaM_grid - OmegaLambda_grid

        DL_values = []

        for z_idx, z in enumerate(self.z):
            print(f"Processing redshift {z_idx + 1}/{len(self.z)}: z = {z:.4f}")

            original_shape = H0_grid.shape

            H0_flat = H0_grid.flatten()
            OmegaM_flat = OmegaM_grid.flatten()
            OmegaK_flat = OmegaK_grid.flatten()
            OmegaLambda_flat = OmegaLambda_grid.flatten()

            DL_flat = np.zeros_like(H0_flat)

            flat_mask = np.abs(OmegaK_flat) < 1e-10
            open_mask = OmegaK_flat > 1e-10
            closed_mask = OmegaK_flat < -1e-10

            print(f"  Flat points: {np.sum(flat_mask)}, Open: {np.sum(open_mask)}, Closed: {np.sum(closed_mask)}")

            # Define a safe integrand that returns 1/sqrt()
            def safe_integrand(OmegaM_subset, OmegaLambda_subset):
                def integrand(z_prime):
                    # Compute the expression from the paper
                    # E(z) = sqrt((1+z)^2(1+OmegaM*z) - z(2+z)*OmegaLambda)
                    term1 = (1 + OmegaM_subset * z_prime) * (1 + z_prime) ** 2
                    term2 = z_prime * (2 + z_prime) * OmegaLambda_subset
                    expr = term1 - term2

                    # Protect against negative or zero values
                    expr = np.maximum(expr, 1e-10)

                    return 1.0 / np.sqrt(expr)

                return integrand

            # Flat universe (OmegaK = 0)
            if np.any(flat_mask):
                try:
                    integrand = safe_integrand(OmegaM_flat[flat_mask], OmegaLambda_flat[flat_mask])
                    comoving_distance, _ = integrate.quad_vec(integrand, 0, z, epsrel=1e-6, limit=100)
                    DL_flat[flat_mask] = c / H0_flat[flat_mask] * (1 + z) * comoving_distance
                except Exception as e:
                    print(f"  Error in flat universe: {e}")
                    DL_flat[flat_mask] = 1e10

            # Open universe (OmegaK > 0)
            if np.any(open_mask):
                try:
                    integrand = safe_integrand(OmegaM_flat[open_mask], OmegaLambda_flat[open_mask])
                    comoving_distance, _ = integrate.quad_vec(integrand, 0, z, epsrel=1e-6, limit=100)
                    sqrt_OmegaK = np.sqrt(OmegaK_flat[open_mask])

                    arg = sqrt_OmegaK * comoving_distance
                    arg = np.clip(arg, -50, 50)  # sinh saturates beyond ~50
                    DL_flat[open_mask] = c / H0_flat[open_mask] * (1 + z) / sqrt_OmegaK * np.sinh(arg)
                except Exception as e:
                    print(f"  Error in open universe: {e}")
                    DL_flat[open_mask] = 1e10

            # Closed universe (OmegaK < 0)
            if np.any(closed_mask):
                try:
                    integrand = safe_integrand(OmegaM_flat[closed_mask], OmegaLambda_flat[closed_mask])
                    comoving_distance, _ = integrate.quad_vec(integrand, 0, z, epsrel=1e-6, limit=100)
                    sqrt_abs_OmegaK = np.sqrt(-OmegaK_flat[closed_mask])
                    arg = sqrt_abs_OmegaK * comoving_distance

                    arg = np.clip(arg, -1e6, 1e6)
                    DL_flat[closed_mask] = c / H0_flat[closed_mask] * (1 + z) / sqrt_abs_OmegaK * np.sin(arg)
                except Exception as e:
                    print(f"  Error in closed universe: {e}")
                    DL_flat[closed_mask] = 1e10

            #invalid values
            DL_flat = np.where(np.isfinite(DL_flat), DL_flat, 1e10)
            DL_flat = np.where(DL_flat > 0, DL_flat, 1e-10)

            # reshape
            DL = DL_flat.reshape(original_shape)
            DL_values.append(DL)


            valid_fraction = np.sum(np.isfinite(DL)) / DL.size
            print(f"  Valid DL fraction: {valid_fraction:.2%}")

        return np.array(DL_values)

    def dist_lum(self, H0, OmegaM, OmegaLambda):

        """
        Get the distance luminosity for every combination of possible cosmological constants

        :param H0: array
        :param OmegaM: array
        :param OmegaLambda: array
        :return: grid of DL
        """

        c = 3e5
        H0_grid, OmegaM_grid, OmegaLambda_grid = np.meshgrid(H0, OmegaM, OmegaLambda, indexing='ij')
        OmegaK_grid = 1 - OmegaM_grid - OmegaLambda_grid

        DL_values = []

        for z in self.z:

            original_shape = H0_grid.shape

            H0_flat = H0_grid.flatten()
            OmegaM_flat = OmegaM_grid.flatten()
            OmegaK_flat = OmegaK_grid.flatten()
            OmegaLambda_flat = OmegaLambda_grid.flatten()

            DL = np.zeros_like(H0_flat)

            flat_mask = np.abs(OmegaK_flat) < 1e-10
            open_mask = OmegaK_flat > 1e-10
            closed_mask = OmegaK_flat < -1e-10

            # Flat universe (OmegaK = 0)
            if np.any(flat_mask):
                def integral_func(z):
                    return 1.0 / np.sqrt(np.abs((1 + OmegaM_flat[flat_mask] * z) * (1 + z) ** 2 - z *
                                         (2 + z) * OmegaLambda_flat[flat_mask]))
                comoving_distance_flat, error_flat = integrate.quad_vec(integral_func, 0, z)
                DL[flat_mask] = c / H0_flat[flat_mask] * (1 + z) * comoving_distance_flat

            # Open universe (OmegaK > 0)
            if np.any(open_mask):
                def integral_func(z):
                    return 1.0 / np.sqrt(np.abs((1 + OmegaM_flat[open_mask] * z) * (1 + z) ** 2 - z *
                                         (2 + z) * OmegaLambda_flat[open_mask]))
                comoving_distance_open, error_open = integrate.quad_vec(integral_func, 0, z)
                sqrt_OmegaK = np.sqrt(OmegaK_flat[open_mask])
                DL[open_mask] = c / H0_flat[open_mask] * (1 + z) / sqrt_OmegaK * np.sinh(sqrt_OmegaK * comoving_distance_open)

            # Closed universe (OmegaK < 0)
            if np.any(closed_mask):
                def integral_func(z):
                    return 1.0 / np.sqrt(np.abs((1 + OmegaM_flat[closed_mask] * z) * (1 + z) ** 2 - z
                                         * (2 + z) * OmegaLambda_flat[closed_mask]))
                comoving_distance_closed, error_closed= integrate.quad_vec(integral_func, 0, z)
                sqrt_abs_OmegaK = np.sqrt(-OmegaK_flat[closed_mask])
                DL[closed_mask] = c / H0_flat[closed_mask] * (1 + z) / sqrt_abs_OmegaK * np.sin(sqrt_abs_OmegaK * comoving_distance_closed)

            DL = DL.reshape(original_shape)
            DL_values.append(DL)

        return np.array(DL_values)

    def calced_dist_mod(self, DL_values):

        """
        Convert the distance luminosity values into distance modulus
        :param DL_values: grid
        :return: grid of mu
        """
        calced_dist_mod = 5 * np.log(DL_values) + 25

        return calced_dist_mod

    def get_chi_squared(self, calced_dist_mod):

        """
        Determine the chi^2 value for every distance modulus
        :param calced_dist_mod: grid
        :return: grid of chi^2
        """

        chi2_grid = np.zeros_like(calced_dist_mod[0])
        for i in range(len(self.z)):
            chiSquared = (calced_dist_mod[i] - self.mu[i]) ** 2 / (self.mu_err[i]**2)
            chi2_grid += chiSquared

        return chi2_grid

    def find_minima(self, grid):

        """
                This function finds the minima in the grid by picking a point in the grid
        at random, then moving to lowest valued neighbour and repeats until there
        are no neighbours with a lower value.

        :param grid:
        :return: the index for the minima chi-squared and the route used to get there
        """

        I, J, K = grid.shape
        i0 = np.random.randint(low=0, high=I)
        j0 = np.random.randint(low=0, high=J)
        k0 = np.random.randint(low=0, high=K)

        def has_lower_neighbor(grid, i0, j0, k0):

            current_val = grid[i0, j0, k0]

            # Check all 26 neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    for dk in [-1, 0, 1]:
                        if di == 0 and dj == 0 and dk == 0:
                            continue  # Skip the point

                        ni, nj, nk = i0 + di, j0 + dj, k0 + dk

                        # Check if neighbor is within bounds
                        if 0 <= ni < I and 0 <= nj < J and 0 <= nk < K:
                            if grid[ni, nj, nk] < current_val:
                                return True  # Found a lower neighbor

            return False  # No lower neighbors found

        def next_step(grid, i0, j0, k0):

            current_val = grid[i0, j0, k0]
            lowest_val = current_val
            best_pos = (i0, j0, k0)

            # Check all 26 neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    for dk in [-1, 0, 1]:
                        if di == 0 and dj == 0 and dk == 0:
                            continue

                        ni, nj, nk = i0 + di, j0 + dj, k0 + dk

                        # Check if neighbor is within bounds
                        if 0 <= ni < I and 0 <= nj < J and 0 <= nk < K:
                            if grid[ni, nj, nk] < lowest_val:
                                lowest_val = grid[ni, nj, nk]
                                best_pos = (ni, nj, nk)

            return best_pos


        path = [(i0, j0, k0)]

        while has_lower_neighbor(grid, i0, j0, k0):
            i0, j0, k0 = next_step(grid, i0, j0, k0)
            path.append((i0, j0, k0))

        return i0, j0, k0, path


    def find_minima2(self, grid):
        """

        :param grid:
        :return:
        """

        print(f"Grid shape: {grid.shape}")
        print(f"Grid size: {grid.size}")
        print(f"Grid dtype: {grid.dtype}")
        print(f"Number of NaN: {np.sum(np.isnan(grid))}")
        print(f"Number of inf: {np.sum(np.isinf(grid))}")
        print(f"Grid min: {np.nanmin(grid) if grid.size > 0 else 'N/A'}")
        print(f"Grid max: {np.nanmax(grid) if grid.size > 0 else 'N/A'}")

        min_pos = ndimage.minimum_position(grid)

        return min_pos

    def plot_chi2(self, chi2_grid):

        print("plotting")
        print(chi2_grid.shape)
        print(chi2_grid[:,0].shape)

        plt.pcolor(chi2_grid[:,0])
        #plt.plot(chi2_grid[:,2])
        plt.show()

        #plt.plot(chi2_grid[:0], chi2_grid[:2])
        #plt.show()

        #plt.plot(chi2_grid[:1], chi2_grid[:2])
        #plt.show()

    def plot_triangle(self, chi2_grid, H0, OmegaM, OmegaLambda):

        # Find best fit
        best_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)

        fig = plt.figure(figsize=(12, 12))

        # H0 vs OmegaM (marginalized over OmegaLambda)
        ax1 = plt.subplot(3, 3, 4)
        chi2_H0_OM = np.min(chi2_grid, axis=2)
        X, Y = np.meshgrid(OmegaM, H0)
        ax1.contourf(X, Y, chi2_H0_OM, levels=20, cmap='Blues')
        ax1.axhline(H0[best_idx[0]], color='r', linestyle='--', alpha=0.5)
        ax1.axvline(OmegaM[best_idx[1]], color='r', linestyle='--', alpha=0.5)
        ax1.set_ylabel('H0')

        # H0 vs OmegaLambda
        ax2 = plt.subplot(3, 3, 7)
        chi2_H0_OL = np.min(chi2_grid, axis=1)
        X, Y = np.meshgrid(OmegaLambda, H0)
        ax2.contourf(X, Y, chi2_H0_OL, levels=20, cmap='Blues')
        ax2.axhline(H0[best_idx[0]], color='r', linestyle='--', alpha=0.5)
        ax2.axvline(OmegaLambda[best_idx[2]], color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('ΩΛ')
        ax2.set_ylabel('H0')

        # OmegaM vs OmegaLambda
        ax3 = plt.subplot(3, 3, 8)
        chi2_OM_OL = np.min(chi2_grid, axis=0)
        X, Y = np.meshgrid(OmegaLambda, OmegaM)
        ax3.contourf(X, Y, chi2_OM_OL, levels=20, cmap='Blues')
        ax3.axhline(OmegaM[best_idx[1]], color='r', linestyle='--', alpha=0.5)
        ax3.axvline(OmegaLambda[best_idx[2]], color='r', linestyle='--', alpha=0.5)
        ax3.plot([0, 1], [1, 0], 'r--', alpha=0.3, label='Flat')
        ax3.set_xlabel('ΩΛ')
        ax3.set_ylabel('ΩM')
        ax3.legend()

        # 1D histograms (marginalized probability distributions)
        ax4 = plt.subplot(3, 3, 1)
        prob_H0 = np.exp(-0.5 * (np.min(chi2_grid, axis=(1, 2)) - np.min(chi2_grid)))
        ax4.plot(prob_H0, H0)
        ax4.set_xlabel('Probability')
        ax4.set_ylabel('H0')

        ax5 = plt.subplot(3, 3, 2)
        prob_OM = np.exp(-0.5 * (np.min(chi2_grid, axis=(0, 2)) - np.min(chi2_grid)))
        ax5.plot(OmegaM, prob_OM)
        ax5.set_xlabel('ΩM')
        ax5.set_ylabel('Probability')

        ax6 = plt.subplot(3, 3, 3)
        prob_OL = np.exp(-0.5 * (np.min(chi2_grid, axis=(0, 1)) - np.min(chi2_grid)))
        ax6.plot(OmegaLambda, prob_OL)
        ax6.set_xlabel('ΩΛ')
        ax6.set_ylabel('Probability')

        plt.tight_layout()
        plt.show()

def chiSquaredFitting():
    df = pd.read_csv('SN_Data.csv')
    SN = df['SN']
    z = df['z']
    mu = df['mu_0']
    mu_err = df['mu_0_error']

    Object = Hubbles(SN, z, mu, mu_err)
    Object.begin_other()

chiSquaredFitting()


