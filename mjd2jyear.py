#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File name: mjd2jyear.py
"""
Created on Wed Feb  5 14:56:40 2020

@author: Neo(liuniu@smail.nju.edu.cn)

MJD -> MJY
"""

import numpy as np
from astropy.time import Time


# -----------------------------  FUNCTIONS -----------------------------
def mjd2jyear(mjd):
    """Convert MJD to MJY.

    Parameter
    ---------
    mjd : 1darray-like
        mean Julian day

    Return
    ------
    mjy : 1darray-like
        mean Julian year
    """

    epoch = Time(mjd, format="mjd")
    jyear = epoch.jyear

    return jyear

# --------------------------------- END --------------------------------
