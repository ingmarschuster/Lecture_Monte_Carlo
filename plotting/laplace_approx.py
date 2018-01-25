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

targd = dist.mixt(1, [dist.mvnorm(np.ones(1)+2, np.ones(1)), dist.mvnorm(np.ones(1)+3.8, np.ones(1))], [0.7, 0.3] )
g = grad(targd.logpdf)
h = hessian(targd.logpdf)
res = sp.optimize.minimize_scalar(lambda x: -targd.logpdf(x))
#maximum = 3.44515
maximum = res['x']
print("Gradient at Maximum logpdf ",g(maximum))
#mpl.style.use('seaborn')

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(x, exp(targd.logpdf(x)), label = 'target density', linewidth = 2)
ax.plot(x, exp(dist.mvnorm(maximum, 1./-h(maximum)).logpdf(x)), '--', label = 'Gaussian approx', linewidth = 2)
ax.legend(loc='best')
ax.set_xticks([])
ax.set_yticks([])
fig.tight_layout()
fig.savefig("../fig/Laplace_approximation_comic.pdf")
