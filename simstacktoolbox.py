import pdb
import os
import shutil
import logging
import pickle
import json
import numpy as np
from astropy.io import fits
from configparser import ConfigParser
from lmfit import Parameters, minimize, fit_report
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from simstackcosmologyestimators import SimstackCosmologyEstimators

from scipy.signal import fftconvolve
# used in smoothing cube layers by psfs

pi = 3.141592653589793
L_sun = 3.839e26  # W
c = 299792458.0  # m/s
conv_lir_to_sfr = 1.728e-10 / 10 ** 0.23
conv_luv_to_sfr = 2.17e-10
a_nu_flux_to_mass = 6.7e19 # erg / s / Hz / Msun
flux_to_specific_luminosity = 1.78  # 1e-23 #1.78e-13
h = 6.62607004e-34  # m2 kg / s  #4.13e-15 #eV/s
k = 1.38064852e-23  # m2 kg s-2 K-1 8.617e-5 #eV/K

class SimstackToolbox(SimstackCosmologyEstimators):

    def __init__(self):
        super().__init__()

    def combine_objects(self, second_object, merge_z=False):
        ''' Combine stacks split into chunks, either by bootstrap or by redshifts.  The second_object is merged into
        self.  If merging redshift slices merge_z=True.

        :param second_object: Simstack object to merge into self.
        :param merge_z: Is True if stacked in redshift chunks.
        :return: second_object merged into self.
        '''

        wavelength_keys = list(self.results_dict['band_results_dict'].keys())
        wavelength_check = list(second_object.results_dict['band_results_dict'].keys())
        if wavelength_keys != wavelength_check:
            "Can't combine these objects. Missing bands"
            pdb.set_trace()

        if merge_z:
            label_dict = self.config_dict['parameter_names']
            label_dict_hi = second_object.config_dict['parameter_names']
            label_dict['redshift'].extend(label_dict_hi['redshift'])
            self.config_dict['catalog']['distance_labels'].extend(second_object.config_dict['catalog']['distance_labels'])
            self.config_dict['distance_bins']['redshift'].extend(second_object.config_dict['distance_bins']['redshift'])
            self.config_dict['distance_bins']['redshift'] = np.unique(
                self.config_dict['distance_bins']['redshift']).tolist()

        for k, key in enumerate(wavelength_keys):
            if merge_z:
                len_results_dict_keys = np.sum(
                    ['flux_densities' in i for i in self.results_dict['band_results_dict'][key].keys()])
                for iboot in np.arange(len_results_dict_keys):
                    if not iboot:
                        boot_label = 'stacked_flux_densities'
                    else:
                        boot_label = 'bootstrap_flux_densities_' + str(int(iboot))

                    self.results_dict['band_results_dict'][key][boot_label].update(
                        second_object.results_dict['band_results_dict'][key][boot_label])
            else:
                self.results_dict['band_results_dict'][key].update(
                    second_object.results_dict['band_results_dict'][key])

    def construct_longname(self, basename):
        ''' Use parameters in config file to create a "longname" which directories and files are named.

        :param basename: customizable "shortname" that proceeds the generic longname.
        :return longname: name used for directory and pickle file.
        '''
        try:
            type_suffix = self.config_dict['catalog']['classification']['split_type']
        except:
            type_suffix = "nuvrj"

        dist_bins = json.loads(self.config_dict['catalog']['classification']['redshift']['bins'])
        dist_suffix = "_".join([str(i).replace('.', 'p') for i in dist_bins]).replace('p0_', '_')
        foreground_suffix = ''
        at_once_suffix = 'layers'
        bootstrap_suffix = ''
        stellar_mass_suffix = ''
        shuffle_suffix = ''
        if 'stellar_mass' in self.config_dict['catalog']['classification']:
            mass_bins = json.loads(self.config_dict['catalog']['classification']['stellar_mass']['bins'])
            stellar_mass_suffix = "_".join(['X', str(len(mass_bins)-1)])
        if 'add_foreground' in self.config_dict['general']['binning']:
            if self.config_dict['general']['binning']['add_foreground']:
                foreground_suffix = 'foregnd'
        if 'stack_all_z_at_once' in self.config_dict['general']['binning']:
            if self.config_dict['general']['binning']['stack_all_z_at_once']:
                at_once_suffix = 'atonce'
        if 'bootstrap' in self.config_dict['general']['error_estimator']:
            if self.config_dict['general']['error_estimator']['bootstrap']['iterations']:
                first_boot = self.config_dict['general']['error_estimator']['bootstrap']['initial_bootstrap']
                last_boot = first_boot + self.config_dict['general']['error_estimator']['bootstrap']['iterations'] - 1
                bootstrap_suffix = "_".join(['bootstrap', "-".join([str(first_boot), str(last_boot)])])
        if 'randomize' in self.config_dict['general']['error_estimator']:
            if self.config_dict['general']['error_estimator']['randomize']:
                shuffle_suffix = 'null'

        if shuffle_suffix == '':
            longname = "_".join([basename, type_suffix, dist_suffix, stellar_mass_suffix, foreground_suffix,
                                 at_once_suffix, bootstrap_suffix])
        else:
            longname = "_".join([basename, type_suffix, dist_suffix, stellar_mass_suffix, foreground_suffix,
                                 at_once_suffix, bootstrap_suffix, shuffle_suffix])
        self.config_dict['io']['longname'] = longname
        return longname

    def copy_config_file(self, fp_in, overwrite_results=False):
        ''' Place copy of config file into longname directory immediately (if you wait it may have been modified before
        stacking is complete)

        :param fp_in: path to config file.
        :param overwrite_results: Overwrite existing if True.
        '''

        if 'shortname' in self.config_dict['io']:
            shortname = self.config_dict['io']['shortname']
        else:
            shortname = os.path.basename(fp_in).split('.')[0]

        longname = self.construct_longname(shortname)

        out_file_path = os.path.join(self.parse_path(self.config_dict['io']['output_folder']), longname)

        if not os.path.exists(out_file_path):
            os.makedirs(out_file_path)
        else:
            if not overwrite_results:
                while os.path.exists(out_file_path):
                    out_file_path = out_file_path + "_"
                os.makedirs(out_file_path)
        self.config_dict['io']['saved_data_path'] = out_file_path

        # Copy Config File
        fp_name = os.path.basename(fp_in)
        fp_out = os.path.join(out_file_path, fp_name)
        logging.info("Copying parameter file...")
        logging.info("  FROM : {}".format(fp_in))
        logging.info("    TO : {}".format(fp_out))
        logging.info("")
        shutil.copyfile(fp_in, fp_out)
        self.config_dict['io']['config_ini'] = fp_out

    def save_stacked_fluxes(self, drop_maps=True, drop_catalogs=False):
        ''' Save pickle containing raw stacked flux results. Optionally drop maps/catalogs to save space.

        :param drop_maps: If True drops maps object.
        :param drop_catalogs: If True drops catalog object.
        :return: Save pickle to saved_data_path in config file.
        '''

        if 'drop_maps' in self.config_dict['io']:
            drop_maps = self.config_dict['io']['drop_maps']
        if 'drop_catalogs' in self.config_dict['io']:
            drop_catalogs = self.config_dict['io']['drop_catalogs']

        longname = self.config_dict['io']['longname']
        out_file_path = self.config_dict['io']['saved_data_path']

        fpath = os.path.join(out_file_path, longname + '.pkl')

        print('pickling to ' + fpath)
        self.config_dict['pickles_path'] = fpath

        # Write simmaps
        if self.config_dict["general"]["error_estimator"]["write_simmaps"] == 1:
            for wv in self.maps_dict:
                name_simmap = wv + '_simmap.fits'
                hdu = fits.PrimaryHDU(self.maps_dict[wv]["flattened_simmap"], header=self.maps_dict[wv]["header"])
                hdul = fits.HDUList([hdu])
                hdul.writeto(os.path.join(out_file_path, name_simmap))
                # self.maps_dict[wv].pop("convolved_layer_cube")
                self.maps_dict[wv].pop("flattened_simmap")

        # Get rid of large files
        if drop_maps:
            print('Removing maps_dict')
            self.maps_dict = {}
        if drop_catalogs:
            print('Removing full_table from catalog_dict')
            self.catalog_dict = {}
            # self.catalog_dict['tables']['full_table'] = {}

        try:
            with open(fpath, "wb") as pickle_file_path:
                pickle.dump(self, pickle_file_path)
        except:
            print("Something wrong with save directory??")
            pdb.set_trace()

        return fpath

    @classmethod
    def import_saved_pickles(cls, pickle_fn):
        with open(pickle_fn, "rb") as file_path:
            encoding = pickle.load(file_path)
        return encoding

    @classmethod
    def save_to_pickles(cls, save_path, save_file):
        with open(save_path, "wb") as pickle_file_path:
            pickle.dump(save_file, pickle_file_path)

    def parse_path(self, path_in):

        path_in = path_in.split(" ")
        if len(path_in) == 1:
            return path_in[0]
        else:
            path_env = os.environ[path_in[0]]
            if len(path_in) == 2:
                if 'nt' in os.name:
                    return path_env + os.path.join('\\', path_in[1].replace('/', '\\'))
                else:
                    return path_env + os.path.join('/', path_in[1])
            else:
                if 'nt' in os.name:
                    path_rename = [i.replace('/', '\\') for i in path_in[1:]]
                    return path_env + os.path.join('\\', *path_rename)
                else:
                    return path_env + os.path.join('/', *path_in[1:])

    def split_bootstrap_labels(self, labels):
        labels_out = []
        for ilabel in labels:
            if ('foreground' in ilabel) or ('background' in ilabel):
                labels_out.append(ilabel)
            else:
                labels_out.append(ilabel)
                labels_out.append(ilabel + '__bootstrap2')

        return labels_out

    def get_params_dict(self, param_file_path):
        config = ConfigParser()
        config.read(param_file_path)

        dict_out = {}
        for section in config.sections():
            dict_sect = {}
            for (each_key, each_val) in config.items(section):
                # Remove quotations from dicts
                try:
                    dict_sect[each_key] = json.loads(each_val)
                except:
                    dict_sect[each_key] = each_val.replace("'", '"')

            dict_out[section] = dict_sect

        # Further remove quotations from embedded dicts
        for dkey in dict_out:
            for vkey in dict_out[dkey]:
                try:
                    dict_out[dkey][vkey] = json.loads(dict_out[dkey][vkey])
                except:
                    pass
        return dict_out

    def write_config_file(self, params_out, config_filename_out):
        config_out = ConfigParser()

        for ikey, idict in params_out.items():
            if not config_out.has_section(ikey):

                config_out.add_section(ikey)
                for isect, ivals in idict.items():
                    # pdb.set_trace()
                    # print(ikey, isect, ivals)
                    config_out.set(ikey, isect, str(ivals))

        # Write config_filename_out (check if overwriting externally)
        with open(config_filename_out, 'w') as conf:
            config_out.write(conf)

    def make_array_from_dict(self, input_dict, x=None):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]
        if x is not None:
            array_out = np.zeros([len(x), *ds])
        else:
            array_out = np.zeros(ds)

        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])
                    label = "__".join([zlab, mlab, plab]).replace('.', 'p')
                    if id_label in input_dict:
                        if x is not None:
                            array_out[:, iz, im, ip] = input_dict[id_label]
                        else:
                            array_out[iz, im, ip] = input_dict[id_label]
                    elif label in input_dict:
                        if x is not None:
                            array_out[:, iz, im, ip] = input_dict[label]
                        else:
                            array_out[iz, im, ip] = input_dict[label]
        return array_out

    def get_80_20_dict(self):
        ''' Get bootstrap flux densities of both the 80 and 20% bins'''
        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]
        bands = list(self.results_dict['band_results_dict'].keys())
        nboots = self.config_dict['general']['error_estimator']['bootstrap']['iterations']

        bootstrap_dict = {}
        outliers_dict = {}

        for iwv in bands:
            bootstrap_matrix = np.zeros([nboots, *ds])
            outliers_matrix = np.zeros([nboots, *ds])
            for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
                for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                    for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                        id_label = "__".join([zlab, mlab, plab])
                        label = "__".join([zlab, mlab, plab]).replace('.', 'p')
                        for ib in range(nboots):
                            blab = 'bootstrap_flux_densities_{0:0.0f}'.format(ib + 1)
                            if label in self.results_dict['band_results_dict'][iwv][blab]:
                                bootstrap_matrix[ib, iz, im, ip] = self.results_dict['band_results_dict'][iwv][blab][
                                    label].value
                                outliers_matrix[ib, iz, im, ip] = self.results_dict['band_results_dict'][iwv][blab][
                                    label + '__bootstrap2'].value

            bootstrap_dict[iwv] = bootstrap_matrix
            outliers_dict[iwv] = outliers_matrix

        bootstrap_full = np.zeros([len(bands), *np.shape(bootstrap_dict[bands[0]])])
        outliers_full = np.zeros([len(bands), *np.shape(bootstrap_dict[bands[0]])])

        for iband, band in enumerate(bands):
            bootstrap_full[iband] = bootstrap_dict[band]
            outliers_full[iband] = outliers_dict[band]

        return_dict = {'bootstrap_dict': bootstrap_dict, 'outliers_dict': outliers_dict,
                       'bootstrap_array': bootstrap_full, 'outliers_array': outliers_full}

        return return_dict

    def get_fast_sed_dict(self, sed_bootstrap_dict):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        x = sed_bootstrap_dict['wavelengths']

        wv_array = self.loggen(8, 1000, 100)
        sed_params_dict = {}
        graybody_dict = {}
        lir_dict = {}
        return_dict = {'wv_array': wv_array, 'sed_params': sed_params_dict, 'graybody': graybody_dict, 'lir': lir_dict}
        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])
                    #label = "__".join([zlab, mlab, plab]).replace('.', 'p')

                    y = sed_bootstrap_dict['sed_fluxes_dict'][id_label]
                    yerr = np.cov(sed_bootstrap_dict['sed_bootstrap_fluxes_dict'][id_label], rowvar=False)
                    zin = sed_bootstrap_dict['z_median'][id_label]

                    sed_params_dict[id_label] = self.fast_sed_fitter(x, y, yerr)
                    graybody_dict[id_label] = self.fast_sed(sed_params_dict[id_label], wv_array)[0]
                    theta = sed_params_dict[id_label]['A'].value, sed_params_dict[id_label]['T_observed'].value
                    lir_dict[id_label] = self.fast_LIR(theta, zin)

        return return_dict

    def get_forced_sed_dict(self, sed_bootstrap_dict):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        x = sed_bootstrap_dict['wavelengths']

        wv_array = self.loggen(8, 1000, 100)
        sed_params_dict = {}
        graybody_dict = {}
        lir_dict = {}
        return_dict = {'wv_array': wv_array, 'sed_params': sed_params_dict, 'graybody': graybody_dict, 'lir': lir_dict}
        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])

                    y = sed_bootstrap_dict['sed_fluxes_dict'][id_label]
                    yerr = np.cov(sed_bootstrap_dict['sed_bootstrap_fluxes_dict'][id_label], rowvar=False)
                    zin = sed_bootstrap_dict['z_median'][id_label]
                    tin = 32.9 + 4.6 * (zin - 2)
                    sed_params_dict[id_label] = self.forced_sed_fitter(x, y, yerr, tin)
                    graybody_dict[id_label] = self.fast_sed(sed_params_dict[id_label], wv_array)[0]
                    theta = sed_params_dict[id_label]['A'].value, sed_params_dict[id_label]['T_observed'].value
                    lir_dict[id_label] = self.fast_LIR(theta, zin)

        return return_dict

    def lambda_to_ghz(self, lam):
        c_light = 299792458.0  # m/s
        return np.array([1e-9 * c_light / (i * 1e-6) for i in lam])

    def graybody_fn(self, theta, x, alphain=2.0, betain=1.8):
        A, T = theta

        c_light = 299792458.0  # m/s

        nu_in = np.array([c_light * 1.e6 / wv for wv in x])
        ng = np.size(A)

        base = 2.0 * (6.626) ** (-2.0 - betain - alphain) * (1.38) ** (3. + betain + alphain) / (2.99792458) ** 2.0
        expo = 34.0 * (2.0 + betain + alphain) - 23.0 * (3.0 + betain + alphain) - 16.0 + 26.0
        K = base * 10.0 ** expo
        w_num = 10 ** A * K * (T * (3.0 + betain + alphain)) ** (3.0 + betain + alphain)
        w_den = (np.exp(3.0 + betain + alphain) - 1.0)
        w_div = w_num / w_den
        nu_cut = (3.0 + betain + alphain) * 0.208367e11 * T
        graybody = np.reshape(10 ** A, (ng, 1)) * nu_in ** np.reshape(betain, (ng, 1)) * self.black(nu_in, T) / 1000.0
        powerlaw = np.reshape(w_div, (ng, 1)) * nu_in ** np.reshape(-1.0 * alphain, (ng, 1))
        graybody[np.where(nu_in >= np.reshape(nu_cut, (ng, 1)))] = \
            powerlaw[np.where(nu_in >= np.reshape(nu_cut, (ng, 1)))]

        return graybody

    def fast_sed(self, m, wavelengths):

        v = m.valuesdict()
        A = np.asarray(v['A'])
        T = np.asarray(v['T_observed'])
        betain = np.asarray(v['beta'])
        alphain = np.asarray(v['alpha'])
        theta_in = A, T

        return self.graybody_fn(theta_in, wavelengths, alphain=alphain, betain=betain)

    def comoving_volume_given_area(self, area_deg2, zz1, zz2):
        vol0 = self.config_dict['cosmology_dict']['cosmology'].comoving_volume(zz2) - \
               self.config_dict['cosmology_dict']['cosmology'].comoving_volume(zz1)
        vol = (area_deg2 / (180. / np.pi) ** 2.) / (4. * np.pi) * vol0
        return vol

    # From Weaver 2022
    def estimate_mlim_70(self, zin):
        return -1.51 * 1e6 * (1 + zin) + 6.81 * 1e7 * (1 + zin) ** 2

    #def weaver_completeness(self, z):
    #    return -3.55 * 1e8 * (1 + z) + 2.70 * 1e8 * (1 + z) ** 2.0

    def moster2011_cosmic_variance(self, z, dz=0.2, field='cosmos'):
        cv_params = {'cosmos': [0.069, -.234, 0.834], 'udf': [0.251, 0.364, 0.358]
            , 'goods': [0.261, 0.854, 0.684], 'gems': [0.161, 0.520, 0.729]
            , 'egs': [0.128, 0.383, 0.673]}

        field_params = cv_params[field]
        sigma_cv_ref = field_params[0] / (z ** field_params[2] + field_params[1])

        if dz == 0.2:
            sigma_cv = sigma_cv_ref
        else:
            sigma_cv = sigma_cv_ref * (dz / 0.2) ** (-0.5)

        return sigma_cv

    def estimate_nuInu(self, wavelength_um, flux_Jy, area_deg2, ngals, completeness=1):
        area_sr = (area_deg2 / (180. / np.pi) ** 2.) / (4. * np.pi)
        return 1e-1 * flux_Jy * (self.lambda_to_ghz(wavelength_um) * 1e9) * 1e-26 * 1e9 / area_sr * ngals / completeness

    def fast_sed_fitter(self, wavelengths, fluxes, covar, betain=1.8, alphain=2.0, redshiftin=0, stellarmassin=None):
        #t_in = (32.9 + 4.6 * (redshiftin - 2)) / (1 + redshiftin)
        t_in = (23.8 + 2.7 * redshiftin + 0.9 * redshiftin ** 2) / (1 + redshiftin)
        if stellarmassin is not None:
            a_in = -47 - redshiftin*0.05 + 11 * (stellarmassin / 10)
        else:
            a_in = -35.0
        fit_params = Parameters()
        fit_params.add('A', value=a_in, vary=True)
        fit_params.add('T_observed', value=t_in, vary=True)
        fit_params.add('beta', value=betain, vary=False)
        fit_params.add('alpha', value=alphain, vary=False)

        fluxin = fluxes #[np.max([i, 1e-7]) for i in fluxes]
        try:
            sed_params = minimize(self.find_sed_min, fit_params,
                                  args=(wavelengths,),
                                  kws={'fluxes': fluxin, 'covar': covar})
            m = sed_params.params
        except:
            m = fit_params

        return m

    def forced_sed_fitter(self, wavelengths, fluxes, covar, t_in, betain=1.8, alphain=2.0):
        a_in = -34.0
        fit_params = Parameters()
        fit_params.add('A', value=a_in, vary=True, max=-32, min=-35)
        fit_params.add('T_observed', value=t_in, vary=False)
        fit_params.add('beta', value=betain, vary=False)
        fit_params.add('alpha', value=alphain, vary=False)

        try:
            sed_params = minimize(self.find_sed_min, fit_params,
                                  args=(wavelengths,),
                                  kws={'fluxes': fluxes, 'covar': covar})
            m = sed_params.params
        except:
            print('NO FIT!')
            m = fit_params

        return m

    def find_sed_min(self, params, wavelengths, fluxes, covar=None):

        graybody = self.fast_sed(params, wavelengths)[0]
        delta_y = (fluxes - graybody)

        if (covar is None) or (np.sum(covar) == 0):
            return delta_y
        else:
            if np.shape(covar) == np.shape(fluxes):
                return delta_y ** 2 / covar
            else:
                return np.matmul(delta_y**2, np.linalg.inv(covar))

    def fast_L850(self, flux850, zin):
        '''This calls graybody_fn instead of fast_sed'''

        conversion = 1 / (1 + zin) * 4.0 * np.pi * (
                self.config_dict['cosmology_dict']['cosmology'].luminosity_distance(
                    zin) * 3.08568025E22) ** 2.0

        Lrf = flux850 * conversion.value * 1e-26 * 1e7  # erg / s / Hz

        return Lrf[0]

    def fast_LIR(self, theta, zin, dzin=None):
        '''This calls graybody_fn instead of fast_sed'''
        wavelength_range = self.loggen(8, 1000, 1000)
        model_sed = self.graybody_fn(theta, wavelength_range)

        nu_in = c * 1.e6 / wavelength_range
        dnu = nu_in[:-1] - nu_in[1:]
        dnu = np.append(dnu[0], dnu)
        Lir = np.sum(model_sed * dnu, axis=1)
        conversion = 4.0 * np.pi * (
                    1.0E-13 * self.config_dict['cosmology_dict']['cosmology'].luminosity_distance(
                zin) * 3.08568025E22) ** 2.0 / L_sun  # 4 * pi * D_L^2    units are L_sun/(Jy x Hz)

        Lrf = (Lir * conversion.value)[0]  # Jy x Hz

        if dzin is not None:
            dLrf = np.zeros([2])
            for idz, dz in enumerate(dzin):
                conversion = 4.0 * np.pi * (
                        1.0E-13 * self.config_dict['cosmology_dict']['cosmology'].luminosity_distance(
                    dz) * 3.08568025E22) ** 2.0 / L_sun  # 4 * pi * D_L^2    units are L_sun/(Jy x Hz)
                dLrf[idz] = (Lir * conversion.value)[0]

            return Lrf, dLrf

        return Lrf

    def fast_Lir(self, m, zin):  # Tin,betain,alphain,z):
        '''I dont know how to do this yet'''
        wavelength_range = self.loggen(8, 1000, 1000)
        model_sed = self.fast_sed(m, wavelength_range)

        nu_in = c * 1.e6 / wavelength_range
        # ns = len(nu_in)
        # dnu = nu_in[0:ns - 1] - nu_in[1:ns]
        dnu = nu_in[:-1] - nu_in[1:]
        dnu = np.append(dnu[0], dnu)
        Lir = np.sum(model_sed * dnu, axis=1)
        conversion = 4.0 * np.pi * (1.0E-13 * self.config_dict['cosmology_dict']['cosmology'].luminosity_distance(
            zin) * 3.08568025E22) ** 2.0 / L_sun  # 4 * pi * D_L^2    units are L_sun/(Jy x Hz)

        Lrf = Lir * conversion  # Jy x Hz
        return Lrf

    def fast_dust_mass(self, lambda_in, flux_in, z_in, T_in, beta=1.8):

        nu_in = c * 1.e6 / lambda_in

        kappa = 0.05 * (lambda_in / 870) ** (beta)  # m^2/kg -- Liang 2019

        Lrf = self.fast_L850(flux_in, z_in)
        Md = 1e-7 * Lrf / kappa / self.blackbody_fn(nu_in, T_in)  # erg/s/Hz  -- Eales 2009

        conversion = 1 / 1.988e30

        return Md[0] * conversion

    def black(self, nu_in, T):
        # h = 6.623e-34     ; Joule*s
        # k = 1.38e-23      ; Joule/K
        # c = 3e8           ; m/s
        # (2*h*nu_in^3/c^2)*(1/( exp(h*nu_in/k*T) - 1 )) * 10^29

        a0 = 1.4718e-21  # 2*h*10^29/c^2
        a1 = 4.7993e-11  # h/k

        num = a0 * nu_in ** 3.0
        den = np.exp(a1 * np.outer(1.0 / T, nu_in)) - 1.0
        ret = num / den

        return ret

    def blackbody_fn(self, nu_in, T_in):
        h = 6.623e-34  # ; Joule*s
        k = 1.38e-23  # ; Joule/K
        c = 2.9979e8  # ; m/s

        return 2 * h * (nu_in ** 3) / c ** 2 / (np.exp(h * nu_in / (k * T_in)) - 1)

    def clean_nans(self, dirty_array, replacement_char=0.0):
        clean_array = dirty_array.copy()
        clean_array[np.isnan(dirty_array)] = replacement_char
        clean_array[np.isinf(dirty_array)] = replacement_char

        return clean_array

    def gauss(self, x, x0, y0, sigma):
        p = [x0, y0, sigma]
        return p[1] * np.exp(-((x - p[0]) / p[2]) ** 2)

    def gauss_kern(self, fwhm, side, pixsize):
        ''' Create a 2D Gaussian (size= side x side)'''

        sig = fwhm / 2.355 / pixsize
        delt = np.zeros([int(side), int(side)])
        delt[0, 0] = 1.0
        ms = np.shape(delt)
        delt = self.shift_twod(delt, ms[0] / 2, ms[1] / 2)
        kern = delt
        gaussian_filter(delt, sig, output=kern)
        kern /= np.max(kern)

        return kern

    def shift_twod(self, seq, x, y):
        out = np.roll(np.roll(seq, int(x), axis=1), int(y), axis=0)
        return out
    
    def smooth_psf(self, mapin, psfin):
        # use scipy function here for speed
        return fftconvolve(mapin, psfin, mode='same')

    def dist_idl(self, n1, m1=None):
        ''' Copy of IDL's dist.pro
        Create a rectangular array in which each element is
        proportinal to its frequency'''

        if m1 == None:
            m1 = int(n1)

        x = np.arange(float(n1))
        for i in range(len(x)): x[i] = min(x[i], (n1 - x[i])) ** 2.

        a = np.zeros([int(n1), int(m1)])

        i2 = m1 // 2 + 1

        for i in range(i2):
            y = np.sqrt(x + i ** 2.)
            a[:, i] = y
            if i != 0:
                a[:, m1 - i] = y

        return a

    def circle_mask(self, pixmap, radius_in, pixres):
        ''' Makes a 2D circular image of zeros and ones'''

        radius = radius_in / pixres
        xy = np.shape(pixmap)
        xx = xy[0]
        yy = xy[1]
        beforex = np.log2(xx)
        beforey = np.log2(yy)
        if beforex != beforey:
            if beforex > beforey:
                before = beforex
            else:
                before = beforey
        else:
            before = beforey
        l2 = np.ceil(before)
        pad_side = int(2.0 ** l2)
        outmap = np.zeros([pad_side, pad_side])
        outmap[:xx, :yy] = pixmap

        dist_array = self.shift_twod(self.dist_idl(pad_side, pad_side), pad_side / 2, pad_side / 2)
        circ = np.zeros([pad_side, pad_side])
        ind_one = np.where(dist_array <= radius)
        circ[ind_one] = 1.
        mask = np.real(np.fft.ifft2(np.fft.fft2(circ) *
                                    np.fft.fft2(outmap))
                       ) * pad_side * pad_side
        mask = np.round(mask)
        ind_holes = np.where(mask >= 1.0)
        mask = mask * 0.
        mask[ind_holes] = 1.
        maskout = self.shift_twod(mask, pad_side / 2, pad_side / 2)

        return maskout[:xx, :yy]

    def map_rms(self, map, mask=None):
        if mask != None:
            ind = np.where((mask == 1) & (self.clean_nans(map) != 0))
            print('using mask')
        else:
            ind = self.clean_nans(map) != 0
        map /= np.max(map)

        x0 = abs(np.percentile(map, 99))
        hist, bin_edges = np.histogram(np.unique(map), range=(-x0, x0), bins=30, density=True)

        p0 = [0., 1., x0 / 3]
        x = .5 * (bin_edges[:-1] + bin_edges[1:])
        x_peak = 1 + np.where((hist - max(hist)) ** 2 < 0.01)[0][0]

        # Fit the data with the function
        fit, tmp = curve_fit(self.gauss, x[:x_peak], hist[:x_peak] / max(hist), p0=p0)
        rms_1sig = abs(fit[2])

        return rms_1sig

    def leja_mass_function(self, z, Mass=np.linspace(9, 13, 100), sfg=2):
        # sfg = 0  -  Quiescent
        # sfg = 1  -  Star Forming
        # sfg = 2  -  All

        nz = np.shape(z)

        a1 = [-0.10, -0.97, -0.39]
        a2 = [-1.69, -1.58, -1.53]
        p1a = [-2.51, -2.88, -2.46]
        p1b = [-0.33, 0.11, 0.07]
        p1c = [-0.07, -0.31, -0.28]
        p2a = [-3.54, -3.48, -3.11]
        p2b = [-2.31, 0.07, -0.18]
        p2c = [0.73, -0.11, -0.03]
        ma = [10.70, 10.67, 10.72]
        mb = [0.00, -0.02, -0.13]
        mc = [0.00, 0.10, 0.11]

        aone = a1[sfg] + np.zeros(nz)
        atwo = a2[sfg] + np.zeros(nz)
        phione = 10 ** (p1a[sfg] + p1b[sfg] * z + p1c[sfg] * z ** 2)
        phitwo = 10 ** (p2a[sfg] + p2b[sfg] * z + p2c[sfg] * z ** 2)
        mstar = ma[sfg] + mb[sfg] * z + mc[sfg] * z ** 2

        # P[0]=alpha, P[1]=M*, P[2]=phi*, P[3]=alpha_2, P[4]=M*_2, P[5]=phi*_2
        P = np.array([aone, mstar, phione, atwo, mstar, phitwo])
        return self.dschecter(Mass, P)

    def davidzon_mass_function(self, z, Mass=np.linspace(9, 13, 100), sfg='sf'):
        # sfg = 0  -  Quiescent
        # sfg = 1  -  Star Forming
        # sfg = 2  -  All

        nz = np.shape(z)

        # logMstar, a1, PhiStar1, a2, PhiStar2
        # P[0]=alpha, P[1]=M*, P[2]=phi*, P[3]=alpha_2, P[4]=M*_2, P[5]=phi*_2
        allg = {
            '0.2_z_0.5': [-1.38, 10.78, 1.187, -0.43, 10.78, 1.92],
            '0.5_z_0.8': [-1.36, 10.77, 1.070, 0.03, 10.77, 1.68],
            '0.8_z_1.1': [-1.31, 10.56, 1.428, 0.51, 10.56, 2.19],
            '1.1_z_1.5': [-1.28, 10.62, 1.069, 0.29, 10.62, 1.21],
            '1.5_z_2.0': [-1.28, 10.51, 0.969, 0.82, 10.51, 0.64],
            '2.0_z_2.5': [-1.57, 10.60, 0.295, 0.07, 10.60, 0.45],
            '2.5_z_3.0': [-1.67, 10.59, 0.228, -0.08, 10.59, 0.21],
            '3.0_z_3.5': [-1.76, 10.83, 0.090, np.nan, 10.83, np.nan],
            '3.5_z_4.5': [-1.98, 11.10, 0.016, np.nan, 11.10, np.nan],
            '4.5_z_5.5': [-2.11, 11.30, 0.003, np.nan, 11.30, np.nan]
        }
        sfg = {
            '0.2_z_0.5': [-1.29, 10.26, 2.410, 1.10, 10.26, 1.30],
            '0.5_z_0.8': [-1.32, 10.40, 1.661, 0.84, 10.40, 0.86],
            '0.8_z_1.1': [-1.29, 10.35, 1.739, 0.81, 10.35, 0.95],
            '1.1_z_1.5': [-1.21, 10.42, 1.542, 1.11, 10.42, 0.49],
            '1.5_z_2.0': [-1.24, 10.40, 1.156, 0.90, 10.40, 0.46],
            '2.0_z_2.5': [-1.50, 10.45, 0.441, 0.59, 10.45, 0.38],
            '2.5_z_3.0': [-1.52, 10.39, 0.441, 1.05, 10.39, 0.13],
            '3.0_z_3.5': [-1.78, 10.83, 0.086, np.nan, 10.83, np.nan],
            '3.5_z_4.0': [-1.84, 10.77, 0.052, np.nan, 10.77, np.nan],
            '4.0_z_6.0': [-2.12, 11.30, 0.003, np.nan, 11.30, np.nan]
        }
        allg = {
            '0.2_z_0.5': [10.78, -1.38, 1.187, -0.43, 1.92],
            '0.5_z_0.8': [10.77, -1.36, 1.070,  0.03, 1.68],
            '0.8_z_1.1': [10.56, -1.31, 1.428,  0.51, 2.19],
            '1.1_z_1.5': [10.62, -1.28, 1.069,  0.29, 1.21],
            '1.5_z_2.0': [10.51, -1.28, 0.969,  0.82, 0.64],
            '2.0_z_2.5': [10.60, -1.57, 0.295,  0.07, 0.45],
            '2.5_z_3.0': [10.59, -1.67, 0.228, -0.08, 0.21],
            '3.0_z_3.5': [10.83, -1.76, 0.090, np.nan, np.nan],
            '3.5_z_4.5': [11.10, -1.98, 0.016, np.nan, np.nan],
            '4.5_z_5.5': [11.30, -2.11, 0.003, np.nan, np.nan]
        }
        sfg = {
            '0.2_z_0.5': [10.26, -1.29, 2.410,  1.10, 1.30],
            '0.5_z_0.8': [10.40, -1.32, 1.661,  0.84, 0.86],
            '0.8_z_1.1': [10.35, -1.29, 1.739,  0.81, 0.95],
            '1.1_z_1.5': [10.42, -1.21, 1.542,  1.11, 0.49],
            '1.5_z_2.0': [10.40, -1.24, 1.156,  0.90, 0.46],
            '2.0_z_2.5': [10.45, -1.50, 0.441,  0.59, 0.38],
            '2.5_z_3.0': [10.39, -1.52, 0.441,  1.05, 0.13],
            '3.0_z_3.5': [10.83, -1.78, 0.086, np.nan, np.nan],
            '3.5_z_4.0': [10.77, -1.84, 0.052, np.nan, np.nan],
            '4.0_z_6.0': [11.30, -2.12, 0.003, np.nan, np.nan]
        }
        qg = {
            '0.2_z_0.5': [10.83, -1.30, 0.098, -0.39, 1.58],
            '0.5_z_0.8': [10.83, -1.46, 0.012, -0.21, 1.44],
            '0.8_z_1.1': [10.75, -0.07, 1.724, np.nan, np.nan],
            '1.1_z_1.5': [10.56,  0.53, 0.757, np.nan, np.nan],
            '1.5_z_2.0': [10.54,  0.93, 0.251, np.nan, np.nan],
            '2.0_z_2.5': [10.69,  0.17, 0.068, np.nan, np.nan],
            '2.5_z_3.0': [10.24,  1.15, 0.028, np.nan, np.nan],
            '3.0_z_3.5': [10.10,  1.15, 0.010, np.nan, np.nan],
            '3.5_z_4.0': [10.10,  1.15, 0.004, np.nan, np.nan]
        }
        SMF = {'sf': sfg, 'qt': qg, 'all': allg}
        smf = SMF[sfg][z]
        aone = a1[sfg] + np.zeros(nz)
        atwo = a2[sfg] + np.zeros(nz)
        phione = 10 ** (p1a[sfg] + p1b[sfg] * z + p1c[sfg] * z ** 2)
        phitwo = 10 ** (p2a[sfg] + p2b[sfg] * z + p2c[sfg] * z ** 2)
        mstar = ma[sfg] + mb[sfg] * z + mc[sfg] * z ** 2

        # P[0]=alpha, P[1]=M*, P[2]=phi*, P[3]=alpha_2, P[4]=M*_2, P[5]=phi*_2

        P = np.array([aone, mstar, phione, atwo, mstar, phitwo])
        return self.dschecter(Mass, P)

    def schecter(self, X, P, exp=None, plaw=None):
        ''' X is alog10(M)
            P[0]=alpha, P[1]=M*, P[2]=phi*
            the output is in units of [Mpc^-3 dex^-1] ???
        '''
        if exp != None:
            return np.log(10.) * P[2] * np.exp(-10 ** (X - P[1]))
        if plaw != None:
            return np.log(10.) * P[2] * (10 ** ((X - P[1]) * (1 + P[0])))
        return np.log(10.) * P[2] * (10. ** ((X - P[1]) * (1.0 + P[0]))) * np.exp(-10. ** (X - P[1]))

    def dschecter(self, X, P):
        '''Fits a double Schechter function but using the same M*
           X is alog10(M)
           P[0]=alpha, P[1]=M*, P[2]=phi*, P[3]=alpha_2, P[4]=M*_2, P[5]=phi*_2
        '''
        rsch1 = np.log(10.) * P[2] * (10. ** ((X - P[1]) * (1 + P[0]))) * np.exp(-10. ** (X - P[1]))
        rsch2 = np.log(10.) * P[5] * (10. ** ((X - P[4]) * (1 + P[3]))) * np.exp(-10. ** (X - P[4]))

        return rsch1 + rsch2

    def get_A_given_z_M_T(self, z, M, Tobs):
        LIR = np.log10(self.sf_main_sequence(z, 10**M) / conv_lir_to_sfr)
        A = np.linspace(-32.5, -36)
        Ldiff = 1e3
        for ai in A:
            Ltemp = np.log10(self.fast_LIR((ai, Tobs / (1 + z)), z))
            if (Ltemp - LIR) ** 2 < Ldiff:
                Ldiff = (Ltemp - LIR) ** 2
                Aout = ai
        return Aout

    def sf_main_sequence(self, z, M):
        ''' From Bethermin 2017, adopted from Schreiber 2015'''

        a0 = 1.5
        a1 = 0.3
        m0 = 0.5
        m1 = 0.36
        a2 = 2.5
        sfr = 10 ** (np.log10(M / 1e9) - m0 + a0 * np.log10(1 + z) - a1 * (
            np.max([0, np.log10(M / 1e9) - m1 - a2 * np.log10(1 + z)])) ** 2)

        return sfr

    def loggen(self, minval, maxval, npoints, linear=None):
        points = np.arange(npoints) / (npoints - 1)
        if (linear != None):
            return (maxval - minval) * points + minval
        else:
            return 10.0 ** ((np.log10(maxval / minval)) * points + np.log10(minval))

    def L_fun(self, p, zed):
        '''Luminosities in log(L)'''
        v = p.valuesdict()
        lum = v["s0"] - (1. + (zed / v["zed0"]) ** (-1.0 * v["gamma"]))
        return lum

    def L_fit(self, p, zed, L, Lerr):
        '''Luminosities in log(L)'''
        lum = self.L_fun(p, zed)
        return (L - lum) / Lerr

    def viero_2013_luminosities(self, z, mass, sfg=1):
        import numpy as np
        y = np.array([[-7.2477881, 3.1599509, -0.13741485],
                      [-1.6335178, 0.33489572, -0.0091072162],
                      [-7.7579780, 1.3741780, -0.061809584]])
        ms = np.shape(y)
        npp = ms[0]
        nz = len(z)
        nm = len(mass)

        ex = np.zeros([nm, nz, npp])
        logl = np.zeros([nm, nz])

        for iz in range(nz):
            for im in range(nm):
                for ij in range(npp):
                    for ik in range(npp):
                        ex[im, iz, ij] += y[ij, ik] * mass[im] ** (ik)
                for ij in range(npp):
                    logl[im, iz] += ex[im, iz, ij] * z[iz] ** (ij)

        T_0 = 27.0
        z_T = 1.0
        epsilon_T = 0.4
        Tdust = T_0 * ((1 + np.array(z)) / (1.0 + z_T)) ** (epsilon_T)

        return [logl, Tdust]

    def viero_2013_luminosities(z, mass, sfg=1):
        import numpy as np
        y = np.array([[-7.2477881, 3.1599509, -0.13741485],
                      [-1.6335178, 0.33489572, -0.0091072162],
                      [-7.7579780, 1.3741780, -0.061809584]])
        ms = np.shape(y)
        npp = ms[0]
        nz = len(z)
        nm = len(mass)

        ex = np.zeros([nm, nz, npp])
        logl = np.zeros([nm, nz])

        for iz in range(nz):
            for im in range(nm):
                for ij in range(npp):
                    for ik in range(npp):
                        ex[im, iz, ij] += y[ij, ik] * mass[im] ** (ik)
                for ij in range(npp):
                    logl[im, iz] += ex[im, iz, ij] * z[iz] ** (ij)

        T_0 = 27.0
        z_T = 1.0
        epsilon_T = 0.4
        Tdust = T_0 * ((1 + np.array(z)) / (1.0 + z_T)) ** (epsilon_T)

        return [logl, Tdust]
