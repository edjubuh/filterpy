from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
from math import cos, sin
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.linalg as linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spln
import scipy.stats
from scipy.stats import norm, multivariate_normal
import warnings

from .stats import *


def plot_discrete_cdf(xs, ys, ax=None, xlabel=None, ylabel=None,
                      label=None):
    """Plots a normal distribution CDF with the given mean and variance.
    x-axis contains the mean, the y-axis shows the cumulative probability.

    Parameters
    ----------

    xs : list-like of scalars
        x values corresponding to the values in `y`s. Can be `None`, in which
        case range(len(ys)) will be used.

    ys : list-like of scalars
        list of probabilities to be plotted which should sum to 1.

    ax : matplotlib axes object, optional
        If provided, the axes to draw on, otherwise plt.gca() is used.

    xlim, ylim: (float,float), optional
        specify the limits for the x or y axis as tuple (low,high).
        If not specified, limits will be automatically chosen to be 'nice'

    xlabel : str,optional
        label for the x-axis

    ylabel : str, optional
        label for the y-axis

    label : str, optional
        label for the legend

    Returns
    -------
        axis of plot
    """
    if ax is None:
        ax = plt.gca()

    if xs is None:
        xs = range(len(ys))
    ys = np.cumsum(ys)
    ax.plot(xs, ys, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_gaussian_cdf(mean=0., variance=1.,
                      ax=None,
                      xlim=None, ylim=(0., 1.),
                      xlabel=None, ylabel=None,
                      label=None):
    """Plots a normal distribution CDF with the given mean and variance.
    x-axis contains the mean, the y-axis shows the cumulative probability.

    Parameters
    ----------

    mean : scalar, default 0.
        mean for the normal distribution.

    variance : scalar, default 0.
        variance for the normal distribution.

    ax : matplotlib axes object, optional
        If provided, the axes to draw on, otherwise plt.gca() is used.

    xlim, ylim: (float,float), optional
        specify the limits for the x or y axis as tuple (low,high).
        If not specified, limits will be automatically chosen to be 'nice'

    xlabel : str,optional
        label for the x-axis

    ylabel : str, optional
        label for the y-axis

    label : str, optional
        label for the legend

    Returns
    -------
        axis of plot
    """
    if ax is None:
        ax = plt.gca()

    sigma = math.sqrt(variance)
    n = scipy.stats.norm(mean, sigma)
    if xlim is None:
        xlim = [n.ppf(0.001), n.ppf(0.999)]

    xs = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0]) / 1000.)
    cdf = n.cdf(xs)
    ax.plot(xs, cdf, label=label)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax



def plot_gaussian_pdf(mean=0., variance=1.,
                      ax=None,
                      mean_line=False,
                      xlim=None, ylim=None,
                      xlabel=None, ylabel=None,
                      label=None):
    """Plots a normal distribution PDF with the given mean and variance.
    x-axis contains the mean, the y-axis shows the probability density.

    Parameters
    ----------

    mean : scalar, default 0.
        mean for the normal distribution.

    variance : scalar, default 0.
        variance for the normal distribution.

    ax : matplotlib axes object, optional
        If provided, the axes to draw on, otherwise plt.gca() is used.

    mean_line : boolean
        draws a line at x=mean

    xlim, ylim: (float,float), optional
        specify the limits for the x or y axis as tuple (low,high).
        If not specified, limits will be automatically chosen to be 'nice'

    xlabel : str,optional
        label for the x-axis

    ylabel : str, optional
        label for the y-axis

    label : str, optional
        label for the legend

    Returns
    -------
        axis of plot
    """

    if ax is None:
        ax = plt.gca()

    sigma = math.sqrt(variance)
    n = scipy.stats.norm(mean, sigma)

    if xlim is None:
        xlim = [n.ppf(0.001), n.ppf(0.999)]

    xs = np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0]) / 1000.)
    ax.plot(xs,n.pdf(xs), label=label)
    ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if mean_line:
        plt.axvline(mean)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax


def plot_gaussian(mean=0., variance=1.,
                  ax=None,
                  mean_line=False,
                  xlim=None,
                  ylim=None,
                  xlabel=None,
                  ylabel=None,
                  label=None):
    """ DEPRECATED. Use plot_gaussian_pdf() instead. This is poorly named, as
    there are multiple ways to plot a Gaussian.

    Plots a normal distribution PDF with the given mean and variance.
    x-axis contains the mean, the y-axis shows the probability density.

    Parameters
    ----------

    ax : matplotlib axes object, optional
        If provided, the axes to draw on, otherwise plt.gca() is used.

    mean_line : boolean
        draws a line at x=mean

    xlim, ylim: (float,float), optional
        specify the limits for the x or y axis as tuple (low,high).
        If not specified, limits will be automatically chosen to be 'nice'

    xlabel : str,optional
        label for the x-axis

    ylabel : str, optional
        label for the y-axis

    label : str, optional
        label for the legend
    """

    warnings.warn('This function is deprecated. It is poorly named. '
                  'A Gaussian can be plotted as a PDF or CDF. This '
                  'plots a PDF. Use plot_gaussian_pdf() instead,',
                  DeprecationWarning)
    return plot_gaussian_pdf(mean, variance, ax, mean_line, xlim, ylim, xlabel,
                             ylabel, label)



def plot_covariance_ellipse(mean, cov=None, variance = 1.0, std=None,
             ellipse=None, title=None, axis_equal=True, show_semiaxis=False,
             facecolor=None, edgecolor=None,
             fc='none', ec='#004080',
             alpha=1.0, xlim=None, ylim=None,
             ls='solid'):
    """ plots the covariance ellipse where

    mean is a (x,y) tuple for the mean of the covariance (center of ellipse)

    cov is a 2x2 covariance matrix.

    `variance` is the normal sigma^2 that we want to plot. If list-like,
    ellipses for all ellipses will be ploted. E.g. [1,2] will plot the
    sigma^2 = 1 and sigma^2 = 2 ellipses. Alternatively, use std for the
    standard deviation, in which case `variance` will be ignored.

    ellipse is a (angle,width,height) tuple containing the angle in radians,
    and width and height radii.

    You may provide either cov or ellipse, but not both.

    plt.show() is not called, allowing you to plot multiple things on the
    same figure.
    """

    assert cov is None or ellipse is None
    assert not (cov is None and ellipse is None)

    if facecolor is None:
        facecolor = fc

    if edgecolor is None:
        edgecolor = ec

    if cov is not None:
        ellipse = covariance_ellipse(cov)

    if axis_equal:
        #plt.gca().set_aspect('equal')
        plt.axis('equal')

    if title is not None:
        plt.title (title)

    compute_std = False
    if std is None:
        std = variance
        compute_std = True


    if np.isscalar(std):
            std = [std]

    if compute_std:
        std = np.sqrt(np.asarray(std))

    ax = plt.gca()

    angle = np.degrees(ellipse[0])
    width = ellipse[1] * 2.
    height = ellipse[2] * 2.

    for sd in std:
        e = Ellipse(xy=mean, width=sd*width, height=sd*height, angle=angle,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                    lw=2, ls=ls)
        ax.add_patch(e)
    x, y = mean
    plt.scatter(x, y, marker='+', color=edgecolor) # mark the center
    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    if show_semiaxis:
        a = ellipse[0]
        h, w = height/4, width/4
        plt.plot([x, x+ h*cos(a+np.pi/2)], [y, y + h*sin(a+np.pi/2)])
        plt.plot([x, x+ w*cos(a)], [y, y + w*sin(a)])


def _do_plot_test():

    from numpy.random import multivariate_normal
    p = np.array([[32, 15],[15., 40.]])

    x,y = multivariate_normal(mean=(0,0), cov=p, size=5000).T
    sd = 2
    a,w,h = covariance_ellipse(p,sd)
    print (np.degrees(a), w, h)

    count = 0
    color=[]
    for i in range(len(x)):
        if _is_inside_ellipse(x[i], y[i], 0, 0, a, w, h):
            color.append('b')
            count += 1
        else:
            color.append('r')
    plt.scatter(x,y,alpha=0.2, c=color)


    plt.axis('equal')

    plot_covariance_ellipse(mean=(0., 0.),
                            cov = p,
                            std=sd,
                            facecolor='none')

    print (count / len(x))


def plot_std_vs_var():
    plt.figure()
    x = (0,0)
    P = np.array([[3,1],[1,3]])
    plot_covariance_ellipse(x, P, std=[1,2,3], facecolor='g', alpha=.2)
    plot_covariance_ellipse(x, P, variance=[1,2,3], facecolor='r', alpha=.5)

