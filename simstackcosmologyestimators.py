import pdb
import os
import shutil
import logging
import emcee
import numpy as np
from astropy.io import fits
from lmfit import Parameters, minimize, fit_report
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit

pi = 3.141592653589793
L_sun = 3.839e26  # W
c = 299792458.0  # m/s
#conv_lir_to_sfr = 1.72e-10
conv_lir_to_sfr = 1.728e-10 / 10 ** 0.23
conv_luv_to_sfr = 2.17e-10
a_nu_flux_to_mass = 6.7e19
flux_to_specific_luminosity = 1.78  # 1e-23 #1.78e-13
h = 6.62607004e-34  # m2 kg / s  #4.13e-15 #eV/s
k = 1.38064852e-23  # m2 kg s-2 K-1 8.617e-5 #eV/K

class SimstackCosmologyEstimators:

    def __init__(self):
        pass

    def log_likelihood(self, theta, x, y, cov):
        y_model = self.graybody_fn(theta, x)
        delta_y = y - y_model[0]
        ll = -0.5 * (np.matmul(delta_y, np.matmul(np.linalg.inv(cov), delta_y)) + len(y) * np.log(2 * np.pi) + np.log(
            np.linalg.det(cov)))
        if not np.isfinite(ll):
            return -np.inf
        return ll

    def log_prior(self, theta, theta0):
        A, T = theta
        A0, T0, sigma_A, sigma_T = theta0
        Amin = -42
        Amax = -26
        Tmin = 1
        Tmax = 32

        if Amin < A < Amax and Tmin < T < Tmax and sigma_A is not None:
            return 0.0

        return -np.inf

    def log_prior_informative(self, theta, theta0):
        A, T = theta
        A0, T0, sigma_A, sigma_T = theta0
        Amin = -42
        Amax = -26
        Tmin = 1
        Tmax = 32
        error_infl = 1.0

        if Amin < A < Amax and Tmin < T < Tmax and sigma_A is not None and sigma_T is not None:
            try:
                lp = -0.5 * (np.sum((10**A - 10**A0) ** 2 / (10**sigma_A * error_infl) ** 2) +
                             np.sum((T - T0) ** 2 / (sigma_T * error_infl) ** 2)) + \
                     np.log(1.0/(np.sqrt(2*np.pi)*(10**sigma_A))) + np.log(1.0/(np.sqrt(2*np.pi)*sigma_T))
            except:
                pdb.set_trace()
            return lp
        return -np.inf

    def log_probability(self, theta, x, y, yerr, theta0):
        lp = self.log_prior_informative(theta, theta0)
        #pdb.set_trace()
        if not np.isfinite(lp):
            return -np.inf
        #print(lp, self.log_likelihood(theta, x, y, yerr))
        return lp + self.log_likelihood(theta, x, y, yerr)

    def mcmc_sed_estimator(self, x, y, yerr, theta, mcmc_iterations=2500, mcmc_discard=25):

        pos = np.array([theta[0], theta[1]]) + 1e-1 * np.random.randn(32, 2)
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability, args=(x, y, yerr, theta)
        )
        sampler.run_mcmc(pos, mcmc_iterations, progress=True)
        flat_samples = sampler.get_chain(discard=mcmc_discard, thin=15, flat=True)

        return flat_samples

    def loop_mcmc_sed_estimator(self, sed_bootstrap_dict, tables, mcmc_iterations=500, mcmc_discard=5):

        id_distance = self.config_dict['catalog']['classification']['redshift']['id']
        id_secondary = self.config_dict['catalog']['classification']['stellar_mass']['id']
        split_table = tables['split_table']
        full_table = tables['full_table']
        bin_keys = list(self.config_dict['parameter_names'].keys())

        wvs = sed_bootstrap_dict['wavelengths']
        mcmc_dict = {}
        y_dict = {}
        yerr_dict = {}
        z_dict = {}
        m_dict = {}
        ngals_dict = {}
        return_dict = {'mcmc_dict': mcmc_dict, 'z_median': z_dict, 'm_median': m_dict,
                       'y': y_dict, 'yerr': yerr_dict, 'ngals': ngals_dict, 'wavelengths': wvs}

        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])
                    label = id_label.replace('.', 'p')
                    # print(id_label)
                    ind_gals = (split_table.redshift == iz) & (split_table.stellar_mass == im) & (
                            split_table.split_params == ip)
                    y = sed_bootstrap_dict['sed_fluxes_dict'][label]
                    yerr = np.cov(sed_bootstrap_dict['sed_bootstrap_fluxes_dict'][label], rowvar=False)
                    y_dict[id_label] = y
                    yerr_dict[id_label] = yerr
                    z_median = np.median(full_table[id_distance][ind_gals])
                    m_median = np.median(full_table[id_secondary][ind_gals])
                    z_dict[id_label] = z_median
                    m_dict[id_label] = m_median
                    ngals_dict[id_label] = np.sum(ind_gals)

                    mcmc_dict[id_label] = self.estimate_mcmc_sed(sed_bootstrap_dict, id_label,
                                                                 z_median=z_median,
                                                                 mcmc_iterations=mcmc_iterations,
                                                                 mcmc_discard=mcmc_discard)

        return return_dict

    def estimate_mcmc_sed(self, sed_bootstrap_dict, id_label, z_median=0,
                          mcmc_iterations=500, mcmc_discard=5):
        label = id_label.replace('.', 'p')
        if not z_median:
            z_label = id_label.split('_')[1:3]
            z_median = np.mean([float(i) for i in z_label])

        x = sed_bootstrap_dict['wavelengths']
        y = sed_bootstrap_dict['sed_fluxes_dict'][label]
        yerr = np.cov(sed_bootstrap_dict['sed_bootstrap_fluxes_dict'][label], rowvar=False)

        sed_params = self.fast_sed_fitter(x, y, yerr)
        graybody = self.fast_sed(sed_params, x)[0]
        delta_y = y - graybody
        med_delta = np.median(y / delta_y)

        Ain = sed_params['A'].value
        Aerr = sed_params['A'].stderr
        Tin = sed_params['T_observed'].value
        Terr = sed_params['T_observed'].stderr

        if Tin is None:
            Tin = (10 ** (1.2 + 0.1 * z_median)) / (1 + z_median)
        if Terr is None:
            Terr = Tin * med_delta
        if Ain is None:
            Ain = -39
        if Aerr is None:
            Aerr = Ain * med_delta

        theta0 = Ain, Tin, Aerr, Terr

        if np.isfinite(np.log(np.linalg.det(yerr))):
            flat_samples = self.mcmc_sed_estimator(x, y, yerr, theta0, mcmc_iterations=mcmc_iterations,
                                              mcmc_discard=mcmc_discard)
        else:
            return -np.inf

        return flat_samples

    def get_lir_from_mcmc_samples(self, mcmc_samples, percentiles=[16, 25, 32, 50, 68, 75, 84]):
        lir_dict = {}
        bin_keys = list(self.config_dict['parameter_names'].keys())

        return_dict = {'lir_dict': lir_dict, 'percentiles': percentiles,
                       'z_median': mcmc_samples['z_median'], 'm_median': mcmc_samples['m_median'],
                       'ngals': mcmc_samples['ngals']}

        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])

                    if type(mcmc_samples['mcmc_dict'][id_label]) is not float:
                        try:
                            mcmc_out = [np.percentile(mcmc_samples['mcmc_dict'][id_label][:, i], percentiles) for i in
                                        range(mcmc_samples['mcmc_dict'][id_label].shape[1])]
                        except:
                            pdb.set_trace()
                        z_median = mcmc_samples['z_median'][id_label]

                        for i, vpercentile in enumerate(percentiles):
                            if id_label not in lir_dict:
                                lir_dict[id_label] = \
                                    {str(vpercentile): np.log10(
                                        self.fast_LIR([mcmc_out[0][i], mcmc_out[1][i]], z_median))}
                            else:
                                lir_dict[id_label][str(vpercentile)] = \
                                    np.log10(self.fast_LIR([mcmc_out[0][i], mcmc_out[1][i]], z_median))

        return return_dict

    def estimate_luminosity_density(self, lir_dict, effective_map_area):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        lird_dict = {}
        percentiles = lir_dict['percentiles']
        results_dict = {'lird_dict': lird_dict, 'percentiles': percentiles, 'effective_area': effective_map_area,
                        'z_median': lir_dict['z_median'], 'm_median': lir_dict['m_median'], 'ngals': lir_dict['ngals']}

        for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
                    id_label = "__".join([zlab, mlab, plab])

                    ngals = lir_dict['ngals'][id_label]
                    zlo = float(zlab.split('_')[-2])
                    zhi = float(zlab.split('_')[-1])

                    # Get median redshift and stellar-mass and completeness-correct
                    z_median = lir_dict['z_median'][id_label]
                    m_median = lir_dict['m_median'][id_label]

                    qcomp = self.estimate_quadri_correction(z_median, m_median)
                    comp = 1
                    if (qcomp > 0.3) and (qcomp < 0.99):
                        comp = qcomp
                        print("z={:0.2f}, m={:0.2f} , {:0.2f}".format(z_median, m_median, comp))

                    if id_label in lir_dict['lir_dict']:
                        for ilir, vlir in enumerate(percentiles):
                            if id_label not in lird_dict:
                                lird_dict[id_label] = \
                                    {str(vlir): np.log10(
                                        self.estimate_lird(10 ** lir_dict['lir_dict'][id_label][str(vlir)],
                                                           ngals, effective_map_area, zlo, zhi, completeness=comp))}
                            else:
                                lird_dict[id_label][str(vlir)] = \
                                    np.log10(
                                        self.estimate_lird(10 ** lir_dict['lir_dict'][id_label][str(vlir)],
                                                           ngals, effective_map_area, zlo, zhi, completeness=comp))

        return results_dict

    def estimate_total_lird_array(self, lird_dict, errors=('25', '75')):
        ''' Estimate Weighted Errors'''
        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]
        lird_array_mid = np.zeros(ds)
        lird_array_hi = np.zeros(ds)
        lird_array_lo = np.zeros(ds)

        for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
                    id_label = "__".join([zlab, mlab, plab])  # .replace('.','p')

                    if id_label in lird_dict['lird_dict']:
                        lird_array_mid[iz, im, ip] = 10 ** lird_dict['lird_dict'][id_label]['50']
                        lird_array_lo[iz, im, ip] = 10 ** lird_dict['lird_dict'][id_label][errors[0]]
                        lird_array_hi[iz, im, ip] = 10 ** lird_dict['lird_dict'][id_label][errors[1]]

        lird_total = np.sum(lird_array_mid[:, :, 1], axis=1) + np.sum(lird_array_mid[:, :, 0], axis=1)
        # lird_error = np.sqrt(np.sum(((10 ** lird_array_hi[:, :, 1] - 10 ** lird_array_lo[:, :, 1]) ** 2), axis=1))
        lird_error = np.sqrt(np.sum(
            (((lird_array_hi[:, :, 1] - lird_array_lo[:, :, 1])) ** 2) * lird_array_mid[:, :, 1],
            axis=1) / np.sum(lird_array_mid[:, :, 1], axis=1))

        return {'lird_array': {'50': lird_array_mid, errors[0]: lird_array_lo, errors[1]: lird_array_hi},
                'lird_total': lird_total, 'lird_total_error': lird_error,
                'sfrd_total': conv_lir_to_sfr * lird_total, 'sfrd_total_error': conv_lir_to_sfr * lird_error}

    def estimate_cib(self, sed_bootstrap_dict, area_deg2):

        bin_keys = list(self.config_dict['parameter_names'].keys())

        nuInu = {}
        wvs = sed_bootstrap_dict['wavelengths']
        cib_dict_out = {'wavelengths': wvs, 'nuInu': nuInu, 'ngals': sed_bootstrap_dict['ngals']}

        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    label = "__".join([zlab, mlab, plab]).replace('.', 'p')
                    if label in sed_bootstrap_dict['sed_fluxes_dict']:
                        y = sed_bootstrap_dict['sed_fluxes_dict'][label]
                        yerr = np.cov(sed_bootstrap_dict['sed_bootstrap_fluxes_dict'][label], rowvar=False)
                        ngals = sed_bootstrap_dict['ngals'][label]
                        nuInu[label] = self.estimate_nuInu(wvs, y, area_deg2, ngals, completeness=1)

        return cib_dict_out

    def estimate_lird(self, lir, ngals, area_deg2, zlo, zhi, completeness=1.0):
        vol = self.comoving_volume_given_area(area_deg2, zlo, zhi)
        return lir * 1e0 * ngals / vol.value / completeness

    def estimate_nuInu(self, wavelength_um, flux_Jy, area_deg2, ngals, completeness=1):
        area_sr = (area_deg2 / (180. / np.pi) ** 2.) / (4. * np.pi)
        return 1e-1 * flux_Jy * (self.lambda_to_ghz(wavelength_um) * 1e9) * 1e-26 * 1e9 / area_sr * ngals / completeness