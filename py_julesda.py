import numpy as np
import netCDF4 as nc
import datetime as dt
import scipy.optimize as spop
import multiprocessing as mp
import itertools as itt
import py_jules as pyj
import os
import shutil as sh
import glob
import subprocess
import sys
import py_calc_soil_params as ssc
# Rewrite this as a class?


class Jules_DA:
    def __init__(self, strt_yr=2009, end_yr=2009, lat=10.75, lon=0.25):
        self.lat = lat
        self.lon = lon
        self.n = 0  # iteration
        self.jules_class = pyj.jules()  # python jules wrapper class
        # extract cci obs
        self.esa_cci_dat = nc.Dataset('/export/cloud/nceo/users/if910917/esa_cci_v03/ghana/esacci_sm_1989_2014_'
                                      'regrid.nc', 'r')  # CCI sm observations
        self.strt_d = dt.datetime(strt_yr, 1, 2, 0, 0)
        self.strt_idx = nc.date2index(self.strt_d, self.esa_cci_dat.variables['time'])
        self.end_d = dt.datetime(end_yr, 12, 31, 0, 0)
        self.end_idx = nc.date2index(self.end_d, self.esa_cci_dat.variables['time'])
        self.lat_idx = np.where(self.esa_cci_dat.variables['lat'][:] == lat)[0][0]
        self.lon_idx = np.where(self.esa_cci_dat.variables['lon'][:] == lon)[0][0]
        self.cci_sm = np.array(self.esa_cci_dat.variables['sm'][self.strt_idx:self.end_idx+1, self.lat_idx,
                               self.lon_idx])  # get soil moisture obs
        # if there is no data at specified time raise an error as data assimilation cannot be completed
        if all(x == self.cci_sm[0] for x in self.cci_sm) is True:
            raise ValueError('No data for this location and time!')
        self.cci_sm[self.cci_sm < 0.] = np.nan
        self.cci_sm_err = np.array(self.esa_cci_dat.variables['sm_uncertainty'][self.strt_idx:self.end_idx+1,
                                   self.lat_idx, self.lon_idx])  # get soil mositure uncertainty
        # if missing sm uncertainties use 20% of the observed value
        if all(x == self.cci_sm_err[0] for x in self.cci_sm_err) is True:
            print 'no uncertainty estimate at:', (lat, lon)
            self.cci_sm_err = 0.2 * self.cci_sm
        else:
            self.cci_sm_err[self.cci_sm_err < 0.] = np.nan
        try:
            np.testing.assert_equal(self.cci_sm/self.cci_sm, self.cci_sm_err/self.cci_sm_err)
        except AssertionError:
            self.cci_sm_err = 0.2 * self.cci_sm
        # Find soil parameters for lat lon and set NML model grid file
        self.latlon_dat = nc.Dataset('../lonlat.regional.nc', 'r')
        self.soil_dat = nc.Dataset('../soil.regional_orig.nc', 'r')
        self.jules_class.model_grid_nml.mapping["jules_model_grid_1_lat_bounds"] = str(lat-0.25)+','+str(lat+0.25)+','
        self.jules_class.model_grid_nml.mapping["jules_model_grid_1_lon_bounds"] = str(lon-0.25)+','+str(lon+0.25)+','
        self.lat_idx = np.where(self.latlon_dat.variables['latitude'][:,0] == lat)[0][0]
        self.lon_idx = np.where(self.latlon_dat.variables['longitude'][0] == lon)[0][0]
        self.b = self.soil_dat.variables['field1381'][self.lat_idx, self.lon_idx]
        self.sathh = self.soil_dat.variables['field342'][self.lat_idx, self.lon_idx]
        self.satcon = self.soil_dat.variables['field333'][self.lat_idx, self.lon_idx]
        self.sm_sat = self.soil_dat.variables['field332'][self.lat_idx, self.lon_idx]
        self.sm_crit = self.soil_dat.variables['field330'][self.lat_idx, self.lon_idx]
        self.sm_wilt = self.soil_dat.variables['field329'][self.lat_idx, self.lon_idx]
        self.hcap = self.soil_dat.variables['field335'][self.lat_idx, self.lon_idx]
        self.hcon = self.soil_dat.variables['field336'][self.lat_idx, self.lon_idx]
        self.albsoil = self.soil_dat.variables['field1395'][self.lat_idx, self.lon_idx]
        self.xb = ssc.calc_SandSilt(self.b, self.sm_sat)  # inital guess to 2 optimised parameters
        self.prior_err = 0.07*self.xb  # previously 0.05
        # Set output dirctory
        self.output_dir = "output/"
        self.steps = []

    def run_jules(self, run_id='gh'):
        """
        Runs JULES changing soil parameters
        :param run_id: id of run as a string
        :return: location of JULES output as string
        """
        self.jules_class.output_nml.mapping["JULES_OUTPUT_1_run_id"] = "'" + run_id + "',"
        self.jules_class.ancillaries_nml.mapping["jules_soil_props_1_const_val"] = \
            str(self.b) + ", " + str(self.sathh) + ", " + str(self.satcon) + ", " + str(self.sm_sat) + \
            ", " + str(self.sm_crit) + ", " + str(self.sm_wilt) + ", " + str(self.hcap) + \
            ", " + str(self.hcon) + ", " + str(self.albsoil) + ","
        self.jules_class.runJules()
        return self.output_dir + "/" + run_id + '.outvars.nc'

    def obs_cost(self, jules_nc):
        """
        Calculates observation cost function between jules and cci obs
        :param jules_nc: files location of JULES netcdf output as string
        :return: cost function value for supplied model output
        """
        jules_dat = nc.Dataset(jules_nc, 'r')
        jules_sm = jules_dat.variables['smcl'][:, 0, 0, 0] / 100.
        obs = self.cci_sm[np.logical_not(np.isnan(self.cci_sm))]
        obs_err = self.cci_sm_err[np.logical_not(np.isnan(self.cci_sm_err))]
        mod = jules_sm[np.logical_not(np.isnan(self.cci_sm))]
        innov = [(obs[i] - mod[i]) ** 2 / (obs_err[i]**2) for i in xrange(len(obs))]
        obs_cost = np.sum(innov)
        return obs_cost

    def cost_b(self, b):
        """
        Calculates the whole cost function when varying the b soil parameter
        :param b: soil parameter b as a list
        :return: Cost function value
        """
        self.b = b[0]
        self.n += 1
        mod_cost = (b - 6.631272)**2 / (0.6 * 6.6313)
        jules_nc = self.run_jules(run_id="iter")
        obs_cost = self.obs_cost(jules_nc)
        ret_val = mod_cost + obs_cost
        self.steps.append([self.n, b[0], ret_val[0]])
        return ret_val

    def cost_sand_silt(self, x0):
        """
        Calculates the whole cost function when varying the _b_smcrit_smwilt soil parameters
        :param x0: vector of soil parameter values as a list
        :return: Cost function value
        """
        sand = x0[0]
        silt = x0[1]
        soil_params = ssc.calc_SoilParam(sand, silt)
        self.b = soil_params[0]
        self.sathh = soil_params[1]
        self.satcon = soil_params[2]
        self.sm_sat = soil_params[3]
        self.sm_crit = soil_params[4]
        self.sm_wilt = soil_params[5]
        self.hcap = soil_params[6]
        self.hcon = soil_params[7]  # <-- including these in DA but not consistent after calculation, is this okay?
        self.n += 1
        innov = [(x0[i] - self.xb[i])**2 / (self.prior_err[i]**2) for i in xrange(len(x0))]
        mod_cost = np.sum(innov)
        jules_nc = self.run_jules(run_id="iter")
        obs_cost = self.obs_cost(jules_nc)
        ret_val = mod_cost + obs_cost
        self.steps.append([self.n, x0, ret_val])
        return ret_val

    def minimize_b(self, b):
        res = spop.minimize(self.cost_b, b, method='nelder-mead', options={'xtol': 1e-1, 'disp': True})
        return res

    def minimize(self, x0):
        res = spop.minimize(self.cost_sand_silt, x0, method='nelder-mead', options={'xtol': 1e-1, 'disp': True})
        return res

    def da_run(self):
        self.jules_class.output_nml.mapping["JULES_OUTPUT_1_output_dir"] = "'"+self.output_dir+"',"
        #self.output_dir = out_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        res = self.minimize(self.xb)
        output = open('da_out_'+str(self.lat)+'_'+str(self.lon)+'.csv', 'w')
        for item in self.steps:
            output.write(str(item).strip("[]") + "\n")
        output.close()
        sh.rmtree(self.output_dir)
        print res.x


def spatial_run_setup(lat_lon):
    """
    Runs JULES for specified lat lon
    :param lat_lon: tuple containing latitude and longitude coordinate
    :return: na
    """
    out_dir = 'output_point_' + str(lat_lon[0]) + '_' + str(lat_lon[1]) + '/'
    run_file = '/home/if910917/qSub_runMe/output_point_' + str(lat_lon[0]) + '_' + str(lat_lon[1]) + '_' +\
               os.getcwd()[-2:] + '.bash'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for file in glob.glob(r'*.nml'):
        sh.copy(file, out_dir)
    os.chdir(out_dir)
    os.makedirs('output')
    try:
        Jules_DA(lat=lat_lon[0], lon=lat_lon[1])
        lines = []
        lines.append('cd '+os.getcwd()+'\n')
        lines.append('module load python/canopy\n')
        lines.append('python ../py_julesda.py da_run '+str(lat_lon[0])+' '+str(lat_lon[1])+'\n')
        f = open(run_file, 'w')
        for line in lines:
            f.write(line)
        f.close()
        os.chdir('../')
    except ValueError:
        os.chdir('../')
        sh.rmtree(out_dir)
        print 'No data at:', lat_lon


def update_soil_nc_all(soil_nc, lonlat_nc):
    """
    Writes updated parameters to soil data file
    :param lat_lon_prod: iter tools instance containing lat lons
    :param soil_nc: location of soil netcdf file to update as string
    :param lonlat_nc: location of corresponding lonlat netcdf file for soil nc file as string
    :return: na
    """
    soil_dat = nc.Dataset(soil_nc, 'a')
    latlon_dat = nc.Dataset(lonlat_nc, 'r')
    for da_out in glob.glob("output_point_*/da_out*.csv"):
        lat = float(da_out.split('_')[2])
        lon = float(da_out.split('_')[3][:-3])
        res_load = open(da_out, 'rb')
        lines = res_load.readlines()[-1].split(',')
        lat_idx = np.where(latlon_dat.variables['latitude'][:, 0] == lat)[0][0]
        lon_idx = np.where(latlon_dat.variables['longitude'][0] == lon)[0][0]
        sand = float(lines[1].strip('array([ '))
        silt = float(lines[2].strip(' ])'))
        soil_params = ssc.calc_SoilParam(sand, silt)
        soil_dat.variables['field1381'][lat_idx, lon_idx] = soil_params[0]
        soil_dat.variables['field342'][lat_idx, lon_idx] = soil_params[1]
        soil_dat.variables['field333'][lat_idx, lon_idx] = soil_params[2]
        soil_dat.variables['field332'][lat_idx, lon_idx] = soil_params[3]
        soil_dat.variables['field330'][lat_idx, lon_idx] = soil_params[4]
        soil_dat.variables['field329'][lat_idx, lon_idx] = soil_params[5]
        soil_dat.variables['field335'][lat_idx, lon_idx] = soil_params[6]
        soil_dat.variables['field336'][lat_idx, lon_idx] = soil_params[7]
    soil_dat.close()
    latlon_dat.close()


if __name__ == "__main__":
    if sys.argv[1] == 'setup_da':
        lats = np.array([4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25])
        lons = np.array([-3.25, -2.75, -2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75])
        # lats = np.array([9.75])
        # lons = np.array([0.75])
        for lat_lon in itt.product(lats, lons):
            spatial_run_setup(lat_lon)
    elif sys.argv[1] == 'da_run':
        jcda = Jules_DA(lat=float(sys.argv[2]), lon=float(sys.argv[3]))
        jcda.da_run()
    elif sys.argv[1] == 'run_forecast':
        print 'running forecast'
        update_soil_nc_all('soil.regional.nc', 'lonlat.regional.nc')
        os.chdir('run_forecast')
        subprocess.call(['python', 'py_jules_run.py'])
        print 'forecast finished'