#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: rms_calc.py
"""
Created on Wed Feb 14 11:02:00 2018

@author: Neo(liuniu@smail.nju.edu.cn)

This script is used for calculating the pre-fit wrms,
post-fit wrms, reduced-chi square, and standard deviation.

3 Mar 2018, Niu : now function 'rms_calc' also computes the standard
                  deviation

"""

import numpy as np
from functools import reduce

__all__ = ["rms_calc"]


# -----------------------------  FUNCTIONS -----------------------------
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


if __name__ == '__main__':
    pass

# --------------------------------- END --------------------------------
