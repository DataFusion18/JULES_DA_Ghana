import numpy as np
import netCDF4 as nc
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import plot_utils as plt_ut
import itertools as itt
import py_calc_soil_params as calc_soil
import os
import sys
import shutil as sh
import glob


class PlotJules:
    def __init__(self, esa_nc='/export/cloud/nceo/users/if910917/esa_cci_v03/ghana/esacci_sm_1989_2014_regrid.nc',
                 soil='soil.regional.nc', **kwargs):
        # extract data from JULES runs
        self.jules_dict = {}
        for key, val in kwargs.items():
            dat, lats, lons, time_var, times, years, sm = plt_ut.extract_jules_nc_vars(val)
            self.jules_dict[str(key)] = {'dat': dat, 'lats': lats, 'lons': lons, 'time_var': time_var, 'times': times,
                                         'years': years, 'sm': sm}
        try:
            self.jd_key = self.jules_dict.keys()[0]
        except IndexError:
            print('Must include a key word arguemnt for JULES output netCDF file location as string')
        self.times = self.jules_dict[self.jd_key]['times']
        self.years = self.jules_dict[self.jd_key]['years']
        self.lats = self.jules_dict[self.jd_key]['lats']
        self.lons = self.jules_dict[self.jd_key]['lons']
        # extract cci obs
        self.esa_cci_dat = nc.Dataset(esa_nc, 'r')  # CCI sm observations
        self.strt_idx = nc.date2index(self.times[0], self.esa_cci_dat.variables['time'])
        self.end_idx = nc.date2index(self.times[-1], self.esa_cci_dat.variables['time'])
        self.lat_idx1 = np.where(self.esa_cci_dat.variables['lat'][:] == self.lats[0])[0][0]
        self.lat_idx2 = np.where(self.esa_cci_dat.variables['lat'][:] == self.lats[-1])[0][0]
        self.lon_idx1 = np.where(self.esa_cci_dat.variables['lon'][:] == self.lons[0])[0][0]
        self.lon_idx2 = np.where(self.esa_cci_dat.variables['lon'][:] == self.lons[-1])[0][0]
        self.cci_sm = np.array(self.esa_cci_dat.variables['sm'][self.strt_idx:self.end_idx+1,
                               self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1])  # get soil moisture obs
        self.cci_sm[self.cci_sm < 0.] = np.nan
        self.cci_sm_err = np.array(self.esa_cci_dat.variables['sm_uncertainty'][self.strt_idx:self.end_idx+1,
                               self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1])  # get soil mositure uncertainty
        self.cci_sm_err[self.cci_sm_err < 0.] = np.nan

        # Find soil parameters for lat lon grid
        self.soil_dat = nc.Dataset(soil, 'r')
        self.latlon_dat = nc.Dataset('lonlat.regional.nc', 'r')
        self.lat_idx1 = np.where(self.latlon_dat.variables['latitude'][:,0] == self.lats[0])[0][0]
        self.lat_idx2 = np.where(self.latlon_dat.variables['latitude'][:,0] == self.lats[-1])[0][0]
        self.lon_idx1 = np.where(self.latlon_dat.variables['longitude'][0] == self.lons[0])[0][0]
        self.lon_idx2 = np.where(self.latlon_dat.variables['longitude'][0] == self.lons[-1])[0][0]
        self.b = self.soil_dat.variables['field1381'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.sathh = self.soil_dat.variables['field342'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.satcon = self.soil_dat.variables['field333'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.sm_sat = self.soil_dat.variables['field332'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.sm_crit = self.soil_dat.variables['field330'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.sm_wilt = self.soil_dat.variables['field329'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.hcap = self.soil_dat.variables['field335'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.hcon = self.soil_dat.variables['field336'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        self.albsoil = self.soil_dat.variables['field1395'][self.lat_idx1:self.lat_idx2+1,
                                                            self.lon_idx1:self.lon_idx2+1]
        # Plotting setup
        sns.set_context('poster', font_scale=1., rc={'lines.linewidth': 2., 'lines.markersize': 3})
        sns.set_style('white')
        self.palette = sns.color_palette('Greys', 8)  # "colorblind", 11)

    def time_plt(self, prior, posterior, ylabel=r'(m$^3$ m$^{-3}$)'):
        """
        Plots a da run for selected point
        :param prior: prior modelled value from JULES as array
        :param posterior: posterior modelled value from JULES as array
        :param lat_idx: lat idx on map
        :param lon_idx: lon idx on map
        :return: fig and axis for plot
        """
        fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.plot(self.times, prior[:], label='prior', color=self.palette[0])
        ax.plot(self.times, posterior[:], label='posterior', color=self.palette[2])
        ax.legend(loc=2)
        plt.axvline(x=self.times[365], linestyle='--')

        ax.set_xticks(self.times, minor=True)
        ax.set_ylabel(ylabel)
        myFmt = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_formatter(myFmt)
        fig.subplots_adjust(hspace=0.3)
        return fig

    def da_point(self, prior, posterior, lat_idx, lon_idx):
        """
        Plots a da run for selected point
        :param prior: prior modelled value from JULES as array
        :param posterior: posterior modelled value from JULES as array
        :param lat_idx: lat idx on map
        :param lon_idx: lon idx on map
        :return: fig and axis for plot
        """
        fig, ax = plt.subplots(nrows=1, ncols=1)

        prior_lab = 'JULES prior'
        post_lab = 'JULES posterior'
        ob_lab = ' ESA observations'
        ax.errorbar(self.times, self.cci_sm[:, lat_idx, lon_idx], fmt='none', yerr=self.cci_sm_err[:, lat_idx, lon_idx], ecolor=self.palette[1])

        ax.plot(self.times, prior[:, lat_idx, lon_idx], label=prior_lab, color=self.palette[3])
        ax.plot(self.times, self.cci_sm[:, lat_idx, lon_idx], 'o', label=ob_lab, color=self.palette[7])
        ax.plot(self.times, posterior[:, lat_idx, lon_idx], label=post_lab, color=self.palette[5])
        title = 'Soil moisture comparison for JULES and ESA CCI at (' + str(self.lats[lat_idx]) + ', ' + \
                str(self.lons[lon_idx]) + ')'
        #title = 'Soil moisture comparison of model and satellite observations in northern Ghana'# + str(self.lats[lat_idx]) + ', ' + \
                #str(self.lons[lon_idx]) + ')'
        ax.set_title(title)
        ax.legend(loc=0, frameon=1)
        plt.axvline(x=self.times[365], linestyle='--', color='k')

        ax.set_xticks(self.times, minor=True)
        ax.set_ylabel(r'Soil moisture (m$^3$ m$^{-3}$)')
        ax.set_xlabel('Year')
        ax.set_xlim([self.times[0], self.times[-1]])
        ax.set_ylim([0.0, 0.45])
        myFmt = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_formatter(myFmt)
        #fig.subplots_adjust(hspace=0.3)
        return fig

    def ft_da_point(self, prior, posterior, lat_idx, lon_idx, modes):
        """
        Plots a da run for selected point
        :param prior: prior modelled value from JULES as array
        :param posterior: posterior modelled value from JULES as array
        :param lat_idx: lat idx on map
        :param lon_idx: lon idx on map
        :return: fig and axis for plot
        """
        fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.plot(self.times, plt_ut.fourier_trans(prior[:, lat_idx, lon_idx], modes), label='prior',
                color=self.palette[0])
        ax.plot(self.times, plt_ut.fourier_trans(posterior[:, lat_idx, lon_idx], modes), label='posterior',
                color=self.palette[2])
        ax.plot(self.times, plt_ut.fourier_trans(self.cci_sm[:, lat_idx, lon_idx], modes, cci=True), 'o', label='CCI',
                color=self.palette[3])
        ax.set_title('Soil moisture comparison for JULES and ESA CCI at (' + str(self.lats[lat_idx]) + ', ' +
                     str(self.lons[lon_idx]) + ')')
        ax.legend(loc=2)
        plt.axvline(x=self.times[365], linestyle='--')
        ax.set_xticks(self.times, minor=True)
        ax.set_ylabel(r'(m$^3$ m$^{-3}$)')
        myFmt = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_formatter(myFmt)
        fig.subplots_adjust(hspace=0.3)
        return fig

    def flat_error(self, modelled_sm, wet_dry='wet', fn_key='weighted', subset='none', axes=None):
        """
        Plots a map of error between CCI obs and JULES for either prior or posterior
        :param modelled_sm: array of modelled soil moisture values
        :param colormap: choice of colormap as string
        :return: figure
        """
        idx = np.arange(len(self.times))
        if wet_dry == 'wet':
            t_idx = np.where([self.times[x].month in [3, 4, 5] for x in idx[365:-365]])[0]
        elif wet_dry =='dry':
            t_idx = np.where([self.times[x].month in [11, 12, 1] for x in idx[365:]])[0]  # was for [12, 1, 2]
        elif wet_dry == 'Jan - Jun':
            t_idx = np.where([self.times[x].month in [1,2,3,4,5,6] for x in idx[365:]])[0]
        elif wet_dry =='Jul - Dec':
            t_idx = np.where([self.times[x].month in [7,8,9,10,11,12] for x in idx[365:]])[0]  # was for [12, 1, 2]
        else:
            t_idx = np.where([self.times[x].month in xrange(1, 13) for x in idx[365:]])[0]

        if subset == 'north':
            rmse_flat = plt_ut.flat_rmse(self.cci_sm[t_idx, 10:, :], modelled_sm[t_idx, 10:, :],
                                       self.cci_sm_err[t_idx, 10:, :], fn_key=fn_key)
        elif subset == 'south':
            rmse_flat = plt_ut.flat_rmse(self.cci_sm[t_idx, :10, :], modelled_sm[t_idx, :10, :],
                                       self.cci_sm_err[t_idx, :10, :], fn_key=fn_key)
        else:
            rmse_flat = plt_ut.flat_rmse(self.cci_sm[t_idx], modelled_sm[t_idx], self.cci_sm_err[t_idx], fn_key=fn_key)
        return rmse_flat

    def map_plt(self, arr, colormap='OrRd', ax=None, v_min=None, v_max=None):
        # draw map
        arr[arr < -900] = np.nan
        m = plt_ut.draw_map(low_lat=self.lats[0]-0.25, high_lat=self.lats[-1]+0.25, low_lon=self.lons[0]-0.25,
                            high_lon=self.lons[-1]+0.25, ax=ax)
        cs = m.imshow(arr, cmap=colormap, interpolation='None', vmin=v_min, vmax=v_max)
        return ax, cs, m

    def map_smcl(self, modelled_sm, times, t_step):
        """
        Plots a map of error between CCI obs and JULES for either prior or posterior
        :param modelled_sm: array of modelled soil moisture values
        :param colormap: choice of colormap as string
        :return: figure
        """
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax, cs, m = self.map_plt(modelled_sm[t_step], colormap='YlGnBu', ax=ax, v_min=0,
                              v_max=0.5)
        d_time = times[t_step]
        str_time = d_time.strftime('%Y/%m/%d')
        # add colorbar.
        cbar = m.colorbar(cs, location='bottom', pad="5%")
        cbar.set_label(r'Soil moisture (m$^{3}~$m$^{-3}$)')
        ax.set_title('JULES modelled soil moisture over Ghana ('+str_time+')')
        clevs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, ]
        cbar.set_ticks(clevs, )
        cbar.ax.set_xticklabels(clevs, rotation=45)
        ret_val = fig
        return ret_val

    def map_error(self, modelled_sm, wet_dry='wet', fn_key='weighted', subset='none', axes=None, title='None'):
        """
        Plots a map of error between CCI obs and JULES for either prior or posterior
        :param modelled_sm: array of modelled soil moisture values
        :param colormap: choice of colormap as string
        :return: figure
        """
        fn_dict = {'weighted': ('OrRd', 0., 10.), 'nrmse': ('OrRd', 0., 1.), 'corr_coeff': ('OrRd', 0., 1.),
                   'hxy': ('RdBu', -0.15, 0.15), 'hxy_err': ('RdBu', -5, 5), 'rmse': ('OrRd', 0., 0.15)}
        if axes is not None:
            ax = axes
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1)
        # plot data
        idx = np.arange(len(self.times))
        if wet_dry == 'wet':
            t_idx = np.where([self.times[x].month in [3, 4, 5] for x in idx[365:-365]])[0]
        elif wet_dry =='dry':
            t_idx = np.where([self.times[x].month in [11, 12, 1] for x in idx[365:]])[0]  # was for [12, 1, 2]
        elif wet_dry == 'Jan - Jun':
            t_idx = np.where([self.times[x].month in [1,2,3,4,5,6] for x in idx[365:]])[0]
        elif wet_dry =='Jul - Dec':
            t_idx = np.where([self.times[x].month in [7,8,9,10,11,12] for x in idx[365:]])[0]  # was for [12, 1, 2]
        else:
            t_idx = np.where([self.times[x].month in xrange(1, 13) for x in idx[365:]])[0]

        if subset == 'north':
            rmse_map = plt_ut.map_rmse(self.cci_sm[t_idx, 10:, :], modelled_sm[t_idx, 10:, :],
                                       self.cci_sm_err[t_idx, 10:, :], fn_key=fn_key)
        elif subset == 'south':
            rmse_map = plt_ut.map_rmse(self.cci_sm[t_idx, :10, :], modelled_sm[t_idx, :10, :],
                                       self.cci_sm_err[t_idx, :10, :], fn_key=fn_key)
        else:
            rmse_map = plt_ut.map_rmse(self.cci_sm[t_idx], modelled_sm[t_idx], self.cci_sm_err[t_idx], fn_key=fn_key)
        ax, cs, m = self.map_plt(rmse_map, colormap=fn_dict[fn_key][0], ax=ax, v_min=fn_dict[fn_key][1],
                              v_max=fn_dict[fn_key][2])
        # add colorbar.
        print np.nanmean(rmse_map)
        if axes is None:
            cbar = m.colorbar(cs, location='bottom', pad="5%")
            clevs = [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15,]
            cbar.set_ticks(clevs, )
            cbar.ax.set_xticklabels(clevs, rotation=45)
            cbar.set_label(r'Bias (m$^3$ m$^{-3}$)')
            if title != 'None':
                ax.set_title(title)
            ret_val = fig
        else:
            ret_val = ax, cs
        return ret_val

    def map_err_subplot(self, prior_t2, posterior_t2, prior_t3, posterior_t3, wet_dry='wet', fn_key='weighted'):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6.65,10))
        ax1, cs = self.map_error(prior_t2, wet_dry=wet_dry, fn_key=fn_key, axes=ax[0,0])
        ax[0,0].set_ylabel(r'No DA')
        ax[0, 0].set_title(r'TAMSAT 2')
        #ax[0,0].set_aspect("auto")
        ax2, cs = self.map_error(posterior_t2, wet_dry=wet_dry, fn_key=fn_key, axes=ax[1, 0])
        ax[1,0].set_ylabel(r'DA')
        #ax[1, 0].set_aspect("auto")
        ax3, cs = self.map_error(prior_t3, wet_dry=wet_dry, fn_key=fn_key, axes=ax[0, 1])
        ax[0, 1].set_title(r'TAMSAT 3')
        #ax[0, 1].set_aspect("auto")
        ax4, cs = self.map_error(posterior_t3, wet_dry=wet_dry, fn_key=fn_key, axes=ax[1, 1])
        #ax[1, 1].set_aspect("auto")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(cs, cax=cbar_ax, label=r'Bias (m$^3$ m$^{-3}$)')
        #plt.tight_layout()
        #plt.suptitle('Comparison between JULES and ESA CCI soil moisture ('+wet_dry+')', fontsize=20)
        plt.suptitle('b) Dry season bias')
        fig.subplots_adjust(wspace= 0.10, hspace=0.10)
        return fig

    def season_err(self, model_sm, wet_dry='wet', axes=None, palet=0, lab='prior', fn_key='rmse', subset=None, line_type='-'):
        idx_date = {}
        if axes is not None:
            ax = axes
            ret_val = ax, idx_date
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ret_val = fig, ax
        if subset == 'north':
            cci_sm = self.cci_sm[:, 10:, :-1]
            model_sm = model_sm[:, 10:, :-1]
            cci_sm_err = self.cci_sm_err[:, 10:, :-1]
        elif subset == 'south':
            cci_sm = self.cci_sm[:, :10, :-1]
            model_sm = model_sm[:, :10, :-1]
            cci_sm_err = self.cci_sm_err[:, :10, :-1]
        else:
            cci_sm = self.cci_sm[:]
            model_sm = model_sm[:]
            cci_sm_err = self.cci_sm_err[:]
        if wet_dry == 'dry':
            for yr in self.years[:-1]:
                t_idx = np.where([self.times[x] in per_delta(dt.datetime(yr, 11, 1), dt.datetime(yr+1, 1, 28))
                                 for x in xrange(len(self.times))])[0]
                idx_date[yr] = np.nanmean(plt_ut.map_rmse(cci_sm[t_idx], model_sm[t_idx], cci_sm_err[t_idx],
                                          fn_key=fn_key))
        elif wet_dry == 'wet':
            for yr in self.years[:]:
                t_idx = np.where([self.times[x] in per_delta(dt.datetime(yr, 3, 1), dt.datetime(yr, 5, 30))
                                 for x in xrange(len(self.times))])[0]
                idx_date[yr] = np.nanmean(plt_ut.map_rmse(cci_sm[t_idx], model_sm[t_idx], cci_sm_err[t_idx],
                                          fn_key=fn_key))
        else:
            for yr in self.years[:]:
                t_idx = np.where([self.times[x] in per_delta(dt.datetime(yr, 1, 1), dt.datetime(yr, 12, 30))
                                 for x in xrange(len(self.times))])[0]
                idx_date[yr] = np.nanmean(plt_ut.map_rmse(cci_sm[t_idx], model_sm[t_idx], cci_sm_err[t_idx],
                                          fn_key=fn_key))
        ax.plot(idx_date.keys(), idx_date.values(), line_type, color=self.palette[palet], label=lab)
        return ret_val

    def season_err_subplot(self, prior_t2, posterior_t2, prior_t3, posterior_t3, wd='wet', title=None,
                           fn_key='rmse', subset=None, axes='None'):
        if axes == 'None':
            fig, ax = plt.subplots(nrows=1, ncols=1,)
            ret_val = fig
        else:
            ax = axes
            ret_val = ax
        ax, idx_date = self.season_err(prior_t2, wet_dry=wd, axes=ax, palet=3, lab='T2 no DA', fn_key=fn_key,
                                       subset=subset, line_type='--')
        ax, idx_date = self.season_err(posterior_t2, wet_dry=wd, axes=ax, palet=7, lab='T2 DA',
                                       fn_key=fn_key, subset=subset, line_type='--')
        ax, idx_date = self.season_err(prior_t3, wet_dry=wd, axes=ax, palet=3, lab ='T3 no DA', fn_key=fn_key,
                                       subset=subset)
        ax, idx_date = self.season_err(posterior_t3, wet_dry=wd, axes=ax, palet=7, lab='T3 DA',
                                       fn_key=fn_key, subset=subset)
        ax.legend(loc=0, frameon=1, prop={'size': 11})
        ax.set_xlim([idx_date.keys()[0], idx_date.keys()[-1]])
        ax.set_ylim([0.0, 0.12])
        ax.set_xticks(np.arange(idx_date.keys()[0], idx_date.keys()[-1]+1))
        ax.set_xticklabels([str(yr) for yr in idx_date.keys()])
        ax.set_title(title)
        ax.set_xlabel('Year')
        ax.set_ylabel(r'Soil Moisture RMSE (m$^3$ m$^{-3}$)')
        return ret_val

    def seas_err_subsubplot(self, prior_t2, posterior_t2, prior_t3, posterior_t3):
        fig, ax = plt.subplots(nrows=2, ncols=2,)
        ax1 = self.season_err_subplot(prior_t2, posterior_t2, prior_t3, posterior_t3, wd='wet', title='',
                           fn_key='rmse', subset='north', axes=ax[0, 0])
        #ax[0,0].set_ylabel(r'Prior')
        #ax[0, 0].set_title(r'Sand')
        ax2 = self.season_err_subplot(prior_t2, posterior_t2, prior_t3, posterior_t3, wd='dry', title='',
                           fn_key='rmse', subset='north', axes=ax[0, 1])
        #ax[0, 1].set_title(r'Silt')
        ax3 = self.season_err_subplot(prior_t2, posterior_t2, prior_t3, posterior_t3, wd='wet', title='',
                           fn_key='rmse', subset='south', axes=ax[1, 0])
        #ax[0, 2].set_title(r'Clay')
        ax4 = self.season_err_subplot(prior_t2, posterior_t2, prior_t3, posterior_t3, wd='dry', title='',
                           fn_key='rmse', subset='south', axes=ax[1, 1])

        #fig.subplots_adjust(right=0.8)
        #fig.subplots_adjust(vspace=.5)
        #plt.suptitle('Comparison of prior and posterior soil maps', fontsize=20)
        #fig.subplots_adjust(wspace= 0.15, hspace=0.001)
        return fig

    def soil_map(self, soil_map, ssc='sand', axes=None):
        if axes is not None:
            ax = axes
        elif axes is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)
        soil_dat = nc.Dataset(soil_map, 'r')
        b = soil_dat.variables['field1381'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        print b.shape
        sm_sat = soil_dat.variables['field332'][self.lat_idx1:self.lat_idx2+1, self.lon_idx1:self.lon_idx2+1]
        print sm_sat.shape
        soil = np.empty_like(b)
        for xy in itt.product(np.arange(soil.shape[0]), np.arange(soil.shape[1])):
            soil[xy[0], xy[1]] = calc_soil.calc_SandSiltClay(b[xy[0], xy[1]], sm_sat[xy[0], xy[1]], ssc=ssc)
        soil = soil * (np.nanmean(self.cci_sm, axis=0)/np.nanmean(self.cci_sm, axis=0))
        ax, cs, m = self.map_plt(soil, colormap='Greys', ax=ax, v_min=0.0, v_max=100.0)
        # add colorbar.
        if axes is None:
            cbar = m.colorbar(cs, location='bottom', pad="5%")
            cbar.set_label('Percentage ' + ssc)
            ret_val = fig
        else:
            ret_val = ax, cs
        return ret_val

    def soil_subplot(self, soil_map_prior, soil_map_posterior):
        fig, ax = plt.subplots(nrows=2, ncols=3,)
        ax1, cs = self.soil_map(soil_map_prior, axes=ax[0,0])
        ax[0,0].set_ylabel(r'Prior')
        ax[0, 0].set_title(r'Sand')
        ax2, cs = self.soil_map(soil_map_prior, ssc='silt', axes=ax[0, 1])
        ax[0, 1].set_title(r'Silt')
        ax3, cs = self.soil_map(soil_map_prior, ssc='clay', axes=ax[0, 2])
        ax[0, 2].set_title(r'Clay')
        ax4, cs = self.soil_map(soil_map_posterior, ssc='sand', axes=ax[1, 0])
        ax[1, 0].set_ylabel(r'Posterior')
        ax5, cs = self.soil_map(soil_map_posterior, ssc='silt', axes=ax[1, 1])
        ax6, cs = self.soil_map(soil_map_posterior, ssc='clay', axes=ax[1, 2])
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(cs, cax=cbar_ax, label=r'Soil texture (%)')
        #plt.suptitle('Comparison of prior and posterior soil maps', fontsize=20)
        #fig.subplots_adjust(wspace= 0.15, hspace=0.001)
        return fig


def per_delta(start, end, delta=dt.timedelta(days=1)):
    dat_arr = np.array([])
    curr = start
    while curr <= end:
        dat_arr = np.append(dat_arr, curr)
        curr += delta
    return dat_arr


def get_plot_class():
    file_base = '/export/cloud/nceo/users/if910917/jules/examples/'
    exp_dir_t3 = 'test_gh_t3/'  # 'soil_flat/da_grid_gh_qsub_err/'  # 'gh_qsub_t3/'
    exp_dir_t2 = 'test_gh_t2/'  # 'gh_qsub_t2/'
    pp = PlotJules(prior_t3_sm=file_base+exp_dir_t3+'run_forecast/output/prior.outvars.nc',
                   posterior_t3_sm=file_base+exp_dir_t3+'run_forecast/output/posterior.outvars.nc',
                   prior_t2_sm=file_base+exp_dir_t2+'run_forecast/output/prior.outvars.nc',
                   posterior_t2_sm=file_base+exp_dir_t2+'run_forecast/output/posterior.outvars.nc')
    return pp


def get_plot_class2():
    file_base = '/export/cloud/nceo/users/if910917/jules/examples/'
    exp_dir_t3 = 'test_gh_7t3/'  # 'soil_flat/da_grid_gh_qsub_err/'  # 'gh_qsub_t3/'
    exp_dir_t2 = 'test_gh_7t2/'  # 'gh_qsub_t2/'
    pp = PlotJules(prior_t3_sm=file_base+exp_dir_t3+'run_forecast/output/prior.outvars.nc',
                   posterior_t3_sm=file_base+exp_dir_t3+'run_forecast/output/posterior.outvars.nc',
                   prior_t2_sm=file_base+exp_dir_t2+'run_forecast/output/prior.outvars.nc',
                   posterior_t2_sm=file_base+exp_dir_t2+'run_forecast/output/posterior.outvars.nc')
    return pp


if __name__ == "__main__":
    out_dir = sys.argv[1]
    file_base = '/export/cloud/nceo/users/if910917/jules/examples/'
    exp_dir_t3 = 'test_gh_10t3/'  # 'soil_flat/da_grid_gh_qsub_err/'  # 'gh_qsub_t3/'
    exp_dir_t2 = 'test_gh_10t2/'  # 'gh_qsub_t2/'
    pp = PlotJules(prior_t3_sm=file_base+exp_dir_t3+'run_forecast/output/prior.outvars.nc',
                   posterior_t3_sm=file_base+exp_dir_t3+'run_forecast/output/posterior.outvars.nc',
                   prior_t2_sm=file_base+exp_dir_t2+'run_forecast/output/prior.outvars.nc',
                   posterior_t2_sm=file_base+exp_dir_t2+'run_forecast/output/posterior.outvars.nc')
    fig = pp.da_point(pp.jules_dict['prior_t3_sm']['sm'], pp.jules_dict['posterior_t3_sm']['sm'], 12, 7)
    fig.savefig(out_dir+'da_point_t3_12_7.png')
    fig = pp.da_point(pp.jules_dict['prior_t3_sm']['sm'], pp.jules_dict['posterior_t3_sm']['sm'], 10, 7)
    fig.savefig(out_dir + 'da_point_t3_10_7.png')
    fig = pp.da_point(pp.jules_dict['prior_t2_sm']['sm'], pp.jules_dict['posterior_t2_sm']['sm'], 12, 7)
    fig.savefig(out_dir + 'da_point_t2_12_7.png')
    fig = pp.da_point(pp.jules_dict['prior_t2_sm']['sm'], pp.jules_dict['posterior_t2_sm']['sm'], 10, 7)
    fig.savefig(out_dir + 'da_point_t2_10_7.png')
    fig = pp.soil_subplot('soil.regional_orig.nc', file_base+exp_dir_t3+'soil.regional.nc')
    fig.savefig(out_dir + 'soil_map_t3.png')
    fig = pp.map_err_subplot(pp.jules_dict['prior_t2_sm']['sm'], pp.jules_dict['posterior_t2_sm']['sm'],
                           pp.jules_dict['prior_t3_sm']['sm'], pp.jules_dict['posterior_t3_sm']['sm'],
                           wet_dry='dry', fn_key='hxy')
    fig.savefig(out_dir + 'hxy_dry.png')
    fig = pp.map_err_subplot(pp.jules_dict['prior_t2_sm']['sm'], pp.jules_dict['posterior_t2_sm']['sm'],
                           pp.jules_dict['prior_t3_sm']['sm'], pp.jules_dict['posterior_t3_sm']['sm'],
                           wet_dry='wet', fn_key='hxy')
    fig.savefig(out_dir + 'hxy_wet.png')
    fig = pp.season_err_subplot(pp.jules_dict['prior_t2_sm']['sm'], pp.jules_dict['posterior_t2_sm']['sm'],
                              pp.jules_dict['prior_t3_sm']['sm'], pp.jules_dict['posterior_t3_sm']['sm'],
                              wet_dry='wet',
                              title='Comparison of JULES and ESA CCI start of season soil moisture over Ghana (wet)')
    fig.savefig(out_dir + 'season_err_wet_gh.png')
    fig = pp.season_err_subplot(pp.jules_dict['prior_t2_sm']['sm'], pp.jules_dict['posterior_t2_sm']['sm'],
                              pp.jules_dict['prior_t3_sm']['sm'], pp.jules_dict['posterior_t3_sm']['sm'],
                              wet_dry='dry',
                              title='Comparison of JULES and ESA CCI start of season soil moisture over Ghana (dry)')
    fig.savefig(out_dir + 'season_err_dry_gh.png')
    fig = pp.season_err_subplot(pp.jules_dict['prior_t2_sm']['sm'], pp.jules_dict['posterior_t2_sm']['sm'],
                              pp.jules_dict['prior_t3_sm']['sm'], pp.jules_dict['posterior_t3_sm']['sm'],
                              wet_dry='all',
                              title='Comparison of JULES and ESA CCI start of season soil moisture over Ghana (all)')
    fig.savefig(out_dir + 'season_err_all_gh.png')