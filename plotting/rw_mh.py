#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:46:59 2018

@author: Ingmar Schuster
"""

from __future__ import division, print_function, absolute_import

import autograd.numpy as np
import scipy as sp

import scipy.stats as stats

from numpy import exp, log, sqrt
from scipy.misc import logsumexp
from autograd import grad, hessian

import distributions as dist

#import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.linspace(-2,8, 1000)
sp.random.seed(2)

targd = dist.mixt(1, [dist.mvnorm(np.ones(1)+1, np.ones(1)), dist.mvnorm(np.ones(1)+3.8, np.ones(1))], [0.8, 0.2] )
q0 = dist.mvnorm(np.ones(1), np.ones(1)*3)
q1 = dist.mvnorm(np.ones(1), np.ones(1)*0.1)

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x, exp(targd.logpdf(x)) / exp(targd.logpdf(x)).max(), label = 'target density', linewidth = 2)
ax.plot(x, exp(q0.logpdf(x))/ exp(q0.logpdf(x)).max()/2, '-.', label = 'q_0(.|1)', linewidth = 2)
ax.plot(x, exp(q1.logpdf(x)) / exp(q1.logpdf(x)).max()/2, '--', label = 'q_1(.|1)', linewidth = 2)
ax.legend(loc='best')
ax.set_xticks([])
ax.set_yticks([])
#fig.tight_layout(pad=1.5)
fig.savefig("../fig/MH_rw_comic.pdf")
