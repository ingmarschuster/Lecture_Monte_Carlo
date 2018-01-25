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

targd = dist.mixt(1, [dist.mvnorm(np.ones(1), np.ones(1)), dist.mvnorm(np.ones(1)+3.8, np.ones(1))], [0.7, 0.3] )
q0 = dist.mvnorm(np.ones(1)+3, np.ones(1)*2)
samps = q0.rvs(20)
lw = targd.logpdf(samps).flatten() - q0.logpdf(samps)
lw = lw - logsumexp(lw)
q1 = dist.mixt(1, [ dist.mvnorm(mu, np.ones(1)) for mu in samps], lw.flatten(), comp_w_in_logspace=True)
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(x, exp(targd.logpdf(x)), label = 'target density', linewidth = 2)
ax.plot(x, exp(q0.logpdf(x)), '-.', label = 'q0', linewidth = 2)
ax.plot(x, exp(q1.logpdf(x)), '--', label = 'q1', linewidth = 2)
ax.legend(loc='best')
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig("../fig/PMC_rw_comic.pdf")
