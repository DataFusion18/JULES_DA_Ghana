#!/home/db903833//dataLand01/enthoughtDistros/epd-7.2-2-rh5-x86_64/bin/python
# !/usr/bin/env python

# core python modules:
import sys
import subprocess
import datetime
from time import strftime
import shutil
# 3rd party modules:
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
# local modules:
from py_julesNML_run import *


class julesAllNML:
    """
    This class is populated by the contents
    of a module which contains templates
    of all the required JULES namelist files

    """

    def __init__(self):
        self.pft_params_nml = pft_params_nml
        self.pft_params_txt = pft_params_txt
        self.nveg_params_nml = nveg_params_nml
        self.nveg_params_txt = nveg_params_txt
        self.crop_params_nml = crop_params_nml
        self.crop_params_txt = crop_params_txt
        self.urban_nml = urban_nml
        self.urban_txt = urban_txt
        self.jules_vegetation_nml = jules_vegetation_nml
        self.jules_vegetation_txt = jules_vegetation_txt
        self.initial_conditions_nml = initial_conditions_nml
        self.initial_conditions_txt = initial_conditions_txt
        self.imogen_nml = imogen_nml
        self.imogen_txt = imogen_txt
        self.fire_nml = fire_nml
        self.fire_txt = fire_txt
        self.jules_snow_nml = jules_snow_nml
        self.jules_snow_txt = jules_snow_txt
        self.model_grid_nml = model_grid_nml
        self.model_grid_txt = model_grid_txt
        self.jules_soil_nml = jules_soil_nml
        self.jules_soil_txt = jules_soil_txt
        self.triffid_params_nml = triffid_params_nml
        self.triffid_params_txt = triffid_params_txt
        self.jules_surface_nml = jules_surface_nml
        self.jules_surface_txt = jules_surface_txt
        self.prescribed_data_nml = prescribed_data_nml
        self.prescribed_data_txt = prescribed_data_txt
        self.jules_radiation_nml = jules_radiation_nml
        self.jules_radiation_txt = jules_radiation_txt
        self.jules_rivers_nml = jules_rivers_nml
        self.jules_rivers_txt = jules_rivers_txt
        self.jules_surface_types_nml = jules_surface_types_nml
        self.jules_surface_types_txt = jules_surface_types_txt
        self.timesteps_nml = timesteps_nml
        self.timesteps_txt = timesteps_txt
        self.ancillaries_nml = ancillaries_nml
        self.ancillaries_txt = ancillaries_txt
        self.jules_soil_biogeochem_nml = jules_soil_biogeochem_nml
        self.jules_soil_biogeochem_txt = jules_soil_biogeochem_txt
        self.jules_hydrology_nml = jules_hydrology_nml
        self.jules_hydrology_txt = jules_hydrology_txt
        self.drive_nml = drive_nml
        self.drive_txt = drive_txt
        self.output_nml = output_nml
        self.output_txt = output_txt

    def writeNML(self):
        self.pft_params_nml.write()
        self.nveg_params_nml.write()
        self.crop_params_nml.write()
        self.urban_nml.write()
        self.jules_vegetation_nml.write()
        self.initial_conditions_nml.write()
        self.imogen_nml.write()
        self.fire_nml.write()
        self.jules_snow_nml.write()
        self.model_grid_nml.write()
        self.jules_soil_nml.write()
        self.triffid_params_nml.write()
        self.jules_surface_nml.write()
        self.prescribed_data_nml.write()
        self.jules_radiation_nml.write()
        self.jules_rivers_nml.write()
        self.jules_surface_types_nml.write()
        self.timesteps_nml.write()
        self.ancillaries_nml.write()
        self.jules_soil_biogeochem_nml.write()
        self.jules_hydrology_nml.write()
        self.drive_nml.write()
        self.output_nml.write()


class jules(julesAllNML):
    def __init__(self, obsFile=None):
        julesAllNML.__init__(self)
        self.jules = '/home/if910917/jules/models/jules4.8/build/bin/jules.exe'

    def runJules(self):
        """
        Write all NML files to disk.
        Run JULES in a subprocess.
        Check output for fatal errors.
        Return stdout and stderr.
        """

        # write all the nml files here so the
        # user doesn't have to remember to...
        self.writeNML()

        # run JULES
        cmd = []
        cmd.append(self.jules)

        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = p.stdout.readlines()
        err = p.stderr.readlines()
        p.wait()

        # catch "fatal" errors
        for line in out:
            if len(line.split()) == 0: continue
            if line.split()[0] == "[FATAL":
                print >> sys.stderr, "*** runJules: caught fatal error in JULES run:"
                print >> sys.stderr, line,
                sys.exit()

            # catch anything in stderr
        if len(err) > 0:
            #print 'Possible error?'
            for line in err:
                print >> sys.stderr, "*** runJules: caught output on stderr in JULES run:"
                print >> sys.stderr, line,
                # sys.exit()

        return (out, err)


if __name__ == "__main__":
    j = jules()
    j.output_nml.mapping["JULES_OUTPUT_1_run_id"] = "'prior',"
    j.ancillaries_nml.mapping["jules_soil_props_1_file"] = " '../soil.regional_origflat.nc',"
    j.runJules()
    j.output_nml.mapping["JULES_OUTPUT_1_run_id"] = "'posterior',"
    j.ancillaries_nml.mapping["jules_soil_props_1_file"] = " '../soil.regional.nc',"
    j.runJules()
