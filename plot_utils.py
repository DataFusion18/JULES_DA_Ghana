# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:26:19 2017

@author: Ewan Pinnington

utility functions for plotting jules output
"""
import matplotlib
import netCDF4 as nc
import numpy as np
import scipy as sp
import datetime as dt
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
#  plt.rcParams['animation.ffmpeg_path'] = '/opt/tools/bin/ffmpeg'
from mpl_toolkits.basemap import Basemap, cm
import itertools as itt
import os
import seaborn as sns


def extract_jules_nc_vars(nc_file):
    """
    Extracts variables from jules output nc file
    :param nc_file: netcdf jules output files location as string
    :return: netcdf data object, latitude values, longitude valuesm netcdf time variable, array of times,
    array of years and soil moisture array
    """
    dat = nc.Dataset(nc_file, 'r')
    lats = dat.variables['latitude'][:, 0]
    lons = dat.variables['longitude'][0]
    time_var = dat.variables['time']
    times = nc.num2date(time_var[:], time_var.units)
    years = np.unique([date.year for date in times])
    sm = dat.variables['smcl'][:, 0, :, :] / 100.
    return dat, lats, lons, time_var, times, years, sm


def extract_jules_nc_vars_subset(nc_file):
    """
    Extracts variables from jules output nc file
    :param nc_file: netcdf jules output files location as string
    :return: netcdf data object, latitude values, longitude valuesm netcdf time variable, array of times,
    array of years and soil moisture array
    """
    dat = nc.Dataset(nc_file, 'r')
    lats = dat.variables['latitude'][:, 0][10:]  # [:-1]
    lons = dat.variables['longitude'][0][:-1]  # [1:8]
    time_var = dat.variables['time']
    times = nc.num2date(time_var[:], time_var.units)
    years = np.unique([date.year for date in times])
    sm = dat.variables['smcl'][:, 0, 10:, :-1] / 100.  # [:, 0, :-1, 1:8] / 100.
    return dat, lats, lons, time_var, times, years, sm


def draw_map(low_lat=4.5, high_lat=12, low_lon=-3.5, high_lon=1.5, ax='None'):
    """
    Creates a cylindrical Basemap instance.
    :param low_lat: lower left lat
    :param high_lat: upper right lat
    :param low_lon: lower left lon
    :param high_lon: upper right lon
    :param ax: axis to create instance for
    :return: Basemap instance
    """
    if ax == 'None':
        m = Basemap(projection='cyl',resolution='i',
                    llcrnrlat=low_lat, urcrnrlat=high_lat,
                    llcrnrlon=low_lon, urcrnrlon=high_lon)
    else:
        m = Basemap(projection='cyl',resolution='i',
                    llcrnrlat=low_lat, urcrnrlat=high_lat,
                    llcrnrlon=low_lon, urcrnrlon=high_lon, ax=ax)
    # draw coastlines, state and country boundaries, edge of map.
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()
    parallels = np.arange(0., 81, 1.)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels, labels=[False, True, True, False], labelstyle='+/-', fontsize=12)
    meridians = np.arange(0., 361., 1.)
    m.drawmeridians(meridians, labels=[True, False, False, True], labelstyle='+/-', fontsize=12)
    return m


def mae(obs, mod, err):
    """
    Calculates the MAE weight by observation std
    :param obs: observations
    :param mod: modelled observations
    :param err: observation error
    :return: weighted RMSE
    """
    obs2 = obs[np.logical_not(np.isnan(obs))]
    mod2 = mod[np.logical_not(np.isnan(obs))]
    err2 = err[np.logical_not(np.isnan(obs))]
    innov = [abs(obs2[i] - mod2[i]) for i in xrange(len(obs2))]
    rmse = np.sum(innov) / len(obs2)
    return rmse


def rmse(obs, mod, err):
    """
    Calculates the RMSE weight by observation std
    :param obs: observations
    :param mod: modelled observations
    :param err: observation error
    :return: weighted RMSE
    """
    obs2 = obs[np.logical_not(np.isnan(obs))]
    mod2 = mod[np.logical_not(np.isnan(obs))]
    err2 = err[np.logical_not(np.isnan(obs))]
    innov = [(obs2[i] - mod2[i])**2 for i in xrange(len(obs2))]
    rmse = np.sqrt(np.sum(innov) / len(obs2))
    return rmse


def nrmse(obs, mod, err):
    """
    Calculates the RMSE weight by observation mean
    :param obs: observations
    :param mod: modelled observations
    :param err: observation error
    :return: weighted RMSE
    """
    obs2 = obs[np.logical_not(np.isnan(obs))]
    mod2 = mod[np.logical_not(np.isnan(obs))]
    err2 = err[np.logical_not(np.isnan(obs))]
    innov = [(obs2[i] - mod2[i])**2 for i in xrange(len(obs2))]
    rmse = np.sqrt(np.sum(innov) / len(obs2))
    nrmse = rmse / np.mean(obs2)
    return nrmse


def weighted_rmse(obs, mod, err):
    """
    Calculates the RMSE weight by observation std
    :param obs: observations
    :param mod: modelled observations
    :param err: observation error
    :return: weighted RMSE
    """
    obs2 = obs[np.logical_not(np.isnan(obs))]
    mod2 = mod[np.logical_not(np.isnan(obs))]
    err2 = err[np.logical_not(np.isnan(obs))]
    innov = [(obs2[i] - mod2[i])**2 / err2[i]**2 for i in xrange(len(obs2))]
    rmse = np.sqrt(np.sum(innov) / len(obs2))
    return rmse


def hxy(obs, mod, err):
    """
    Calculates the hx - y weighted by observation std
    :param obs: observations
    :param mod: modelled observations
    :param err: observation error
    :return: weighted RMSE
    """
    obs2 = obs[np.logical_not(np.isnan(obs))]
    mod2 = mod[np.logical_not(np.isnan(obs))]
    err2 = err[np.logical_not(np.isnan(obs))]
    innov = [(mod2[i] - obs2[i]) for i in xrange(len(obs2))]
    hxy = np.mean(innov)
    return hxy


def hxy_err(obs, mod, err):
    """
    Calculates the hx - y weighted by observation std
    :param obs: observations
    :param mod: modelled observations
    :param err: observation error
    :return: weighted RMSE
    """
    obs2 = obs[np.logical_not(np.isnan(obs))]
    mod2 = mod[np.logical_not(np.isnan(obs))]
    err2 = err[np.logical_not(np.isnan(obs))]
    innov = [(mod2[i] - obs2[i]) / err2[i] for i in xrange(len(obs2))]
    hxy = np.mean(innov)
    return hxy


def corr_coeff(obs, mod, err):
    """
    Calculates the hx - y weighted by observation std
    :param obs: observations
    :param mod: modelled observations
    :param err: observation error
    :return: weighted RMSE
    """
    obs2 = obs[np.logical_not(np.isnan(obs))]
    mod2 = mod[np.logical_not(np.isnan(obs))]
    err2 = err[np.logical_not(np.isnan(obs))]
    corrc = sp.stats.linregress(obs2, mod2)[2]
    return corrc


def flat_rmse(obs, mod, err, fn_key='rmse'):
    """
    Calculates spatial rmse weighted by observation std for given arrays
    :param obs: observations array with dimensions (time, lat, lon)
    :param mod: modelled observations with dimensions (time, lat, lon)
    :param err: observation errors with dimensions (time, lat, lon)
    :return: weight RMSE array
    """
    fn_dict = {'rmse': rmse, 'nrmse': nrmse, 'weighted': weighted_rmse, 'hxy': hxy, 'hxy_err': hxy_err, 'mae': mae,
               'corr_coeff': corr_coeff}
    ret_val = fn_dict[fn_key](obs, mod, err)
    return ret_val


def map_rmse(obs, mod, err, fn_key='rmse'):
    """
    Calculates spatial rmse weighted by observation std for given arrays
    :param obs: observations array with dimensions (time, lat, lon)
    :param mod: modelled observations with dimensions (time, lat, lon)
    :param err: observation errors with dimensions (time, lat, lon)
    :return: weight RMSE array
    """
    fn_dict = {'rmse': rmse, 'nrmse': nrmse, 'weighted': weighted_rmse, 'hxy': hxy, 'hxy_err': hxy_err, 'mae': mae,
               'corr_coeff': corr_coeff}
    rmse_arr = np.empty_like(mod[0])
    for xy in itt.product(np.arange(rmse_arr.shape[0]), np.arange(rmse_arr.shape[1])):
        rmse_arr[xy[0], xy[1]] = fn_dict[fn_key](obs[:, xy[0], xy[1]], mod[:, xy[0], xy[1]], err[:, xy[0], xy[1]])
    print np.nanmax(rmse_arr), np.nanmin(rmse_arr)
    return rmse_arr


def fourier_trans(dat, modes, cci=False):
    """
    Calculates fft of data, then removes modes above given point and returns inverse ft
    :param dat: data to ft
    :param modes: amount of modes to remove
    :param cci: is the data from esa cci with gaps
    :return: inverse ft
    """
    if cci is True:
        cv_i = np.arange(len(dat))
        mask_cv = np.isfinite(dat)
        dat = np.interp(cv_i, cv_i[mask_cv], dat[mask_cv])
    dat_ft = np.fft.fft(dat)
    dat_ft[modes:] = 0
    dat_ift = np.fft.ifft(dat_ft)
    return dat_ift