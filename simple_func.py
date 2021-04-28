#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: simple_func.py
"""
Created on Tue Apr 20 11:18:07 2021

@author: Neo(niu.liu@nju.edu.cn)
"""

from functools import reduce
import numpy as np
from scipy.special import gammainc


# -----------------------------  FUNCTIONS -----------------------------
def wgt_mean_calc(x, err=None):
    """Calculate the weighted mean and formal error(std) of x series.

    Parameters
    ----------
    x : array, float
        Series
    err : array, float, default is None.

    Returns
    ----------
    mean : float
        (weighted) mean

    std : float
        (weighted) standard deviation
    """

    if err is None:
        mean = np.mean(x)
        std = np.sqrt(np.dot(xn, xn) / (xn.size - 1))
    else:
        p = 1. / err
        mean = np.dot(x, p**2) / np.dot(p, p)
        xn = (x - mean) * p
        std = np.sqrt(np.dot(xn, xn) / np.dot(p, p) / (len(x)-1))

    return mean, std


def rms_calc(x, err=None):
    """Calculate the (weighted) wrms and std of x series.

    Standard deviation
    std = sqrt(sum( (xi-mean)^2/erri^2 ) / sum( 1.0/erri^2 ))
         if weighted,
         = sqrt(sum( (xi-mean)^2/erri^2 ) / (N-1))
         otherwise.

    Weighted root mean square
    wrms = sqrt(sum( xi^2/erri^2 ) / sum( 1.0/erri^2 ))
         if weighted,
         = sqrt(sum( xi^2/erri^2 ) / (N-1))
         otherwise.

    Parameters
    ----------
    x : array, float
        Series
    err : array, float, default is None.

    Returns
    ----------
    mean : float
        (weighted) mean
    wrms : float
        (weighted) rms
    std : float
        (weighted) standard deviation
    """

    if err is None:
        mean = np.mean(x)
        xn = x - mean
        std = np.sqrt(np.dot(xn, xn) / (xn.size - 1))
        wrms = np.sqrt(np.dot(x, x) / (x.size - 1))
    else:
        p = 1. / err
        mean = np.dot(x, p**2) / np.dot(p, p)
        xn = (x - mean) * p
        std = np.sqrt(np.dot(xn, xn) / np.dot(p, p))
        wrms = np.sqrt(np.dot(x*p, x*p) / np.dot(p, p))

    return mean, wrms, std


def calc_chi2(x, err, reduced=False, deg=0):
    '''Calculate the (reduced) Chi-square.


    Parameters
    ----------
    x : array, float
        residuals
    err : array, float
        formal errors of residuals
    reduced : boolean
        True for calculating the reduced chi-square
    deg : int
        degree of freedom

    Returns
    ----------
    (reduced) chi-square
    '''

    wx = x / err
    chi2 = np.dot(wx, wx)

    if reduced:
        if deg:
            return chi2 / (x.size - deg)
        else:
            print("# ERROR: the degree of freedom cannot be 0!")
    else:
        return chi2


def calc_2dchi2(x, errx, y, erry, covxy, reduced=False):
    '''Calculate the 2-Dimension (reduced) Chi-square.


    Parameters
    ----------
    x : array, float
        residuals of x
    errx : array, float
        formal errors of x
    x : array, float
        residuals of x
    errx : array, float
        formal errors of x
    covxy : array, float
        summation of covariance between x and y
    reduced : boolean
        True for calculating the reduced chi-square

    Returns
    ----------
    (reduced) chi-square
    '''

    Qxy = np.zeros_like(x)

    for i, (xi, errxi, yi, erryi, covxyi
            ) in enumerate(zip(x, errx, y, erry, covxy)):

        wgt = np.linalg.inv(np.array([[errxi**2, covxyi],
                                      [covxyi, erryi**2]]))

        Xmat = np.array([xi, yi])

        Qxy[i] = reduce(np.dot, (Xmat, wgt, Xmat))

        if Qxy[i] < 0:
            print(Qxy[i])

    if reduced:
        return np.sum(Qxy) / (x.size - 2)
    else:
        return np.sum(Qxy)


def gof(obs_num, fdm_num, chi2):
    """Calculate the goodness-of-fit.

    The formula is expressed as below.

    Q = gammq((obs_num - fdm_num) / 2, chi2 / 2). (Numerical Recipes)

    gammq is the incomplete gamma function.

    Parameters
    ----------
    fdm_num : int
        number of freedom
    chi2 : float
        chi square

    Return
    ------
    Q : float
        goodness-of-fit
    """

    Q = gammainc((obs_num - fdm_num)/2., chi2/2.)

    return Q


if __name__ == "__main__":
    pass

# --------------------------------- END --------------------------------
