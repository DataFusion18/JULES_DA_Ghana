#!/usr/bin/python
"""This module Is used to calculated soil parameters and sand silt clay fraction
"""
import numpy as np


def calc_SoilParam(sand, silt):
    """
    use soil texture percentages of silt and sand to return soil parameter values
    for use in JULES ancillaries.nml, clay percentage is not given as an input and is calculated based on sand and silt
    input values
    :param sand: percentage sand
    :param silt: percentage silt
    :return: array of JULES soil parameters
    """
    clay = 100 - (sand + silt)

    psis = 0.01 * (10 ** (1.54 - 0.0095 * sand + 0.0063 * silt))
    thetas = (50.5 - 0.142 * sand - 0.037 * clay) / 100
    b = 3.1 + 0.157 * clay - 0.003 * sand
    Ksat_in_hr = 10 ** (-0.6 - 0.0064 * clay + 0.0126 * sand)
    Ksat_m_s = (0.0254 / 3600) * Ksat_in_hr
    FC = thetas * (3.3 / psis) ** (-1 / b)
    WP = thetas * (150 / psis) ** (-1 / b)
    hcap = (1 - thetas) * 1942000
    satcon = Ksat_m_s * 1000
    hcon = -1 * (-0.51 + (thetas * 0.56))
    sathh = psis
    sm_sat = thetas
    sm_crit = FC
    sm_wilt = WP
    albsoil = 0.17  # 0.11 <-- was previously 0.11, query this?

    soil_params = np.array([b, sathh, satcon, sm_sat, sm_crit, sm_wilt, hcap, hcon, albsoil])

    return soil_params


def calc_SandSilt(b, sm_sat):
    """
    Calculates the percentage of soil that is sand/silt
    :param b: brooks and corey exponent factor
    :param sm_sat: saturated point for soil moisture
    :return: sand percentage and silt percentage in numpy array
    """
    sand = - (40.*(78500.*sm_sat + 185.*b - 40216)) / 4481.
    clay = (b - 3.1 + 0.003*sand) / 0.157
    silt = 100 - (sand + clay)

    return np.array([sand, silt])


def calc_SandSiltClay(b, sm_sat, ssc='sand'):
    """
    Calculates the percentage of soil that is sand/silt
    :param b: brooks and corey exponent factor
    :param sm_sat: saturated point for soil moisture
    :param ssc: choice of return val as string of (sand, silt, clay)
    :return: requested value (sand silt clay) as percentage (0-100)
    """
    sand = - (40.*(78500.*sm_sat + 185.*b - 40216)) / 4481.
    clay = (b - 3.1 + 0.003*sand) / 0.157
    silt = 100 - (sand + clay)
    if ssc == 'sand':
        ret_val = sand
    elif ssc == 'silt':
        ret_val = silt
    elif ssc == 'clay':
        ret_val = clay
    return ret_val