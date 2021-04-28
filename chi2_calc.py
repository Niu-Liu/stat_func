#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: chi2_calc.py
"""
Created on Thu Mar 28 10:52:46 2019

@author: Neo(liuniu@smail.nju.edu.cn)
"""

import numpy as np


__all__ = ["calc_chi2", "calc_2dchi2"]


# -----------------------------  FUNCTIONS -----------------------------
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

# -------------------------------- MAIN --------------------------------


# --------------------------------- END --------------------------------
