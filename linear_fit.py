#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: linear_fit.py
"""
Created on Thu Mar 28 10:40:19 2019

@author: Neo(liuniu@smail.nju.edu.cn)
"""


import numpy as np
from numpy import sqrt, sin, cos
from scipy.optimize import curve_fit

__all__ = ["linfit", "linfit2d"]


# -----------------------------  FUNCTIONS -----------------------------
def linear_func(t, x0, x1):
    return x0 + x1 * t


def linfit(x, y, yerr=None):
    """(Weighted) Linear fitting of y(x) = offset + drift * x

    Parameters
    ---------
    x / y : series

    Returns
    -------
    x0/x0_err : float
        estimate and formal uncertainty of offset
    x1/x1_err : float
        estimate and formal uncertainty of drift
    predict : array_like of float
        predicition series from linear model of y
    residual : array_like of float
        residual series from linear model of y
    chi2_dof : float
        reduced chi-square
    """

    if yerr is None:
        popt, pcov = curve_fit(linear_func, x, y)
    else:
        popt, pcov = curve_fit(
            linear_func, x, y, sigma=yerr, absolute_sigma=False)

    # Create a `dict` object to store the results
    res = {}

    # Estimate, `x0` for interception and `x1` for slope
    res["x0"], res["x1"] = popt

    # Formal error and correlation coefficient
    res["x0_err"], res["x1_err"] = sqrt(pcov[0, 0]), sqrt(pcov[1, 1])
    res["cor"] = pcov[0, 1] / offset_err / drift_err

    # Prediction
    res["predict"] = linear_func(x, *popt)

    # Residual
    res["residual"] = y - res["predict"]

    # Chi squared per dof
    if yerr is None:
        res["chi2_dof"] = np.sum(res["residual"]) / (len(y) - 2)
    else:
        res["chi2_dof"] = np.sum(res["residual"] / yerr) / (len(y) - 2)

    return res


# Written on 2021-04-19
# NB: The convention is quite different from what I wrote before.
def linear_func2d(t, x0, x1, y0, y1):
    """A simple 2D linear function.
    """

    x = x1 * t + x0
    y = y1 * t + y0

    return np.concatenate((x, y))


def linear_func2d_tot(t, x0, y0, pm, pa):
    """Another simple 2D linear function.

    Here the total linear drift is estimated rather than two separated components.

    PA is counted eastward from the declination axis.
    """

    x = pm * sin(pa) * t + x0
    y = pm * cos(pa) * t + y0

    return np.concatenate((x, y))


def linfit2d(t, x, y, x_err=None, y_err=None, xy_cor=None, fit_type="sep"):
    """(Weighted) Linear fitting of

    (x,y) = offset + drift * t

    Parameters
    ---------
    x / y : series

    Returns
    -------
    x0, y0/x0_err, y0_err : float
        estimate and formal uncertainty of offset for x, y
    x1, y1/x1_err, y1_err : float
        estimate and formal uncertainty of drift for x, y
    predict_x, predict_y : array_like of float
        predicition series from linear model of x, y
    residual_x, residual_y : array_like of float
        residual series from linear model of x, y
    chi2_dof : float
        reduced chi-square
    """

    # Create a `dict` object to store the results
    res = {}

    # Position vector
    pos = np.concatenate((x, y))
    N = len(t)

    # Covariance matrix
    if x_err is not None:
        cov_mat = np.diag(np.concatenate((x_err ** 2, y_err**2)))

        if xy_cor is not None:

            # Avoid the +/-1 co
            xy_cor = np.where(xy_cor == 1, 0.999999, xy_cor)
            xy_cor = np.where(xy_cor == -1, -0.999999, xy_cor)

            # Consider the correlation between x and y
            xy_cov = x_err * y_err * xy_cor

            for i, xy_covi in enumerate(xy_cov):
                cov_mat[i, i+N] = xy_covi
                cov_mat[i+N, i] = xy_covi

    if fit_type == "sep":
        if x_err is None:
            popt, pcov = curve_fit(linear_func2d, t, pos,
                                   bounds=([-np.inf, -np.inf, -np.inf, -np.inf],
                                           [np.inf, np.inf, np.inf, np.inf]))
        else:
            popt, pcov = curve_fit(linear_func2d, t, pos,
                                   sigma=cov_mat, absolute_sigma=False)

        # Estimate, `x0` for interception and `x1` for slope in x
        res["x0"], res["x1"], res["y0"], res["y1"] = popt

        # Formal error and correlation coefficient
        res["x0_err"], res["x1_err"] = sqrt(pcov[0, 0]), sqrt(pcov[1, 1])
        res["y0_err"], res["y1_err"] = sqrt(pcov[2, 2]), sqrt(pcov[3, 3])
        res["x0x1_cor"] = pcov[0, 1] / res["x0_err"] / res["x1_err"]
        res["y0y1_cor"] = pcov[2, 3] / res["y0_err"] / res["y1_err"]
        res["x1y1_cor"] = pcov[1, 3] / res["x1_err"] / res["y1_err"]

    elif fit_type == "tot":
        popt, pcov = curve_fit(linear_func2d_tot, t, pos,
                               sigma=cov_mat, absolute_sigma=False,
                               bounds=([-np.inf, -np.inf, 0, 0],
                                       [np.inf, np.inf, 2*np.pi]))

        # Estimate, `x0` for interception and `x1` for slope in x
        res["x0"], res["y0"], res["pm"], res["pa"] = popt

        # Formal error and correlation coefficient
        res["x0_err"], res["y0_err"] = sqrt(pcov[0, 0]), sqrt(pcov[1, 1])
        res["pm_err"], res["pa_err"] = sqrt(pcov[2, 2]), sqrt(pcov[3, 3])
        res["pmpa_cor"] = pcov[2, 3] / res["pm_err"] / res["pa_err"]

        # Radian -> degree
        res["pa"] = np.rad2deg(res["pa"])
        res["pa_err"] = np.rad2deg(res["pa_err"])

    else:
        print("Please pass a proper value to fit_type which could only be "
              "'sep' or 'tot'.")
        os.exit(1)

    # Prediction
    predict = linear_func2d(t, *popt)
    res["predict_x"] = predict[:N]
    res["predict_y"] = predict[N:]

    # Residual
    res["residual_x"] = x - res["predict_x"]
    res["residual_y"] = y - res["predict_y"]

    # Chi squared per dof
    if x_err is None:
        res["chi2_dof_x"] = np.sum(res["residual_x"]) / (N - 2)
        res["chi2_dof_y"] = np.sum(res["residual_y"]) / (N - 2)
    else:
        res["chi2_dof_x"] = np.sum(res["residual_x"] / x_err) / (N - 2)
        res["chi2_dof_y"] = np.sum(res["residual_y"] / y_err) / (N - 2)

    return res


# --------------------------------- END --------------------------------
