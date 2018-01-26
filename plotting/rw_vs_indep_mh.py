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
sp.random.seed(5)

targd = dist.mixt(1, [dist.mvnorm(np.ones(1)+1, np.ones(1)), dist.mvnorm(np.ones(1)+3.8, np.ones(1))], [0.8, 0.2] )

qrw = dist.mvnorm(np.zeros(1), np.ones(1)*0.4)
qind = dist.mvnorm(np.ones(1) * 2, np.ones(1)*0.2)

fig, ax = plt.subplots(figsize=(4, 2))
ax.plot(x, exp(targd.logpdf(x)) / exp(targd.logpdf(x)).max(), label = r'$\pi$', linewidth = 2)
for i in range(3):
    current = targd.rvs().flatten()
    ax.plot(x, exp(qrw.logpdf(x-current))/ exp(qrw.logpdf(x-current)).max()/2, '-.', label = r'$q(\cdot|X_%d=%.1f)$' %(i, current), linewidth = 2)
(handles, labels) = ax.get_legend_handles_labels()
ax.legend(handles[:3], labels[:3], loc='best')
ax.set_xticks([])
ax.set_yticks([])
#fig.tight_layout(pad=1.5)
fig.savefig("../fig/MH_rw_comic.pdf")
fig, ax = plt.subplots(figsize=(4, 2))
ax.plot(x, exp(targd.logpdf(x)) / exp(targd.logpdf(x)).max(), label = r'$\pi$', linewidth = 2)
ax.plot(x, exp(qind.logpdf(x/2))/ exp(qind.logpdf(x/2)).max()/2, '-.', label = r'$q(\cdot|X_i), i \in \mathbb{N}$' , linewidth = 2)
ax.legend(loc='best')
ax.set_xticks([])
ax.set_yticks([])
#fig.tight_layout(pad=1.5)
fig.savefig("../fig/MH_indep_comic.pdf")

fig, ax = plt.subplots(figsize=(4, 2))
ax.plot(x, exp(targd.logpdf(x)) / exp(targd.logpdf(x)).max(), label = r'$\pi$', linewidth = 2)
qconv = dist.mixt(1, [dist.mvnorm(targd.comp_dist[0].mu, targd.comp_dist[0].K +qrw.K.flatten()), dist.mvnorm(targd.comp_dist[1].mu, targd.comp_dist[1].K +qrw.K.flatten())], [0.8, 0.2] )
ax.plot(x, exp(qconv.logpdf(x))/ exp(qconv.logpdf(x)).max(), '-.', label = r"$\mathbb{E}_\pi [ q(x'|x) ]$" , linewidth = 2)
ax.legend(loc='best')
ax.set_xticks([])
ax.set_yticks([])
#fig.tight_layout(pad=1.5)
fig.savefig("../fig/MH_marg_comic.pdf")

