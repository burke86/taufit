import os, sys
import warnings
import celerite
from celerite import terms
import emcee
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.timeseries import LombScargle
from scipy.optimize import minimize, differential_evolution

def simulate_drw(t_rest, tau=50, SFinf=0.3, ymean=0, z=0.0, seed=None):
    """
    Simulate DRW given input times, tau, SF, and ymean
    
    t_rest: time (rest frame)
    tau: DRW timescale
    SFinf: structure function at infinity
    ymean: data mean
    z: redshift
    seed: random number generator seed
    
    (Code adapted from astroML)
        Xmean = b * tau
        SFinf = sigma * sqrt(tau / 2)
    """

    N = len(t_rest)

    t_obs = t_rest*(1 + z)/tau
    np.random.seed(seed)
    x = np.zeros(N)
    x[0] = np.random.normal(ymean, SFinf)
    E = np.random.normal(0, 1, N)

    for i in range(1, N):
        dt = t_obs[i] - t_obs[i - 1]
        x[i] = (x[i - 1]
                - dt * (x[i - 1] - ymean)
                + np.sqrt(2) * SFinf * E[i] * np.sqrt(dt))
        
    return x
        
    
def fit_drw(x, y, yerr, init='minimize', nburn=500, nsamp=2000, bounds='default', jitter=False, color="#ff7f0e", plot=True, verbose=True):
    """
    Fit DRW model using celerite
    
    x: time [astropy unit quantity]
    y: data [astropy unit quantity]
    yerr: error on data [astropy unit quantity]
    init: 'minimize', 'differential_evolution', or array of user-specified (e.g. previous) initial conditions
    nburn: number of burn-in samples
    nsamp: number of production samples
    bounds: 'dafault', 'none', or user-specified dictionary following celerite convention
    jitter: whether to add jitter (white noise term)
    color: color for plotting
    plot: whether to plot the result
    verbose: whether to print useful messages
    """
        
    # Sort data
    ind = np.argsort(x)
    x = x[ind]; y = y[ind]; yerr = yerr[ind]
    baseline = x[-1]-x[0]
    
    # Combine Lomb-Scargle periodigram first to validate inputs
    freqLS, powerLS = LombScargle(x, y, yerr).autopower(normalization='psd')
    
    # Use uniform prior with default 'smart' bounds:
    if bounds == 'default':
        min_precision = np.min(yerr.value)
        amplitude = np.max(y.value)-np.min(y.value)
        amin = np.log(0.1*min_precision)
        amax = np.log(10*amplitude)
        log_a = np.mean([amin,amax])
        
        min_cadence = np.clip(np.min(np.diff(x.value)), 0.1, None)
        cmin = np.log(1/(10*baseline.value))
        cmax = np.log(1/min_cadence)
        log_c = np.mean([cmin,cmax])
    elif bounds == 'none':
        amin = -np.inf
        amax = np.inf
        cmin = -np.inf
        cmax = np.inf
        log_a = 0
        log_c= 0
    elif isinstance(bounds, dict):
        pass
    else:
        raise ValueError('bounds value not recognized!')
                
    kernel = terms.RealTerm(log_a=log_a, log_c=log_c,
                            bounds=dict(log_a=(amin, amax), log_c=(cmin, cmax)))
    
    # Add jitter term
    if jitter:
        kernel += terms.JitterTerm(log_sigma=1.0, bounds=dict(log_sigma=(0.1*amin, amax)))
        
    gp, samples = fit_celerite(x, y, yerr, kernel, init=init, nburn=nburn, nsamp=nsamp, color=color, plot=plot, verbose=verbose)
    
    # Return the GP model and sample chains
    return gp, samples

def fit_carma(x, y, yerr, p=2, init='minimize', nburn=500, nsamp=2000, bounds='default', jitter=False, color="#ff7f0e", plot=True, verbose=True):
    """
    Fit CARMA-equivilant model using celerite
    
    Note: x, y, and yerr must by astropy Quantities with units!
    x: time [astropy unit quantity]
    y: data [astropy unit quantity]
    yerr: error on data [astropy unit quantity]
    p: AR order of CARMA model (q = p - 1)
    init: 'minimize', 'differential_evolution', or array of user-specified (e.g. previous) initial conditions
    nburn: number of burn-in samples
    nsamp: number of production samples
    bounds: 'dafault', 'none', or user-specified dictionary following celerite convention
    jitter: whether to add jitter (white noise term)
    color: color for plotting
    plot: whether to plot the result
    verbose: whether to print useful messages
    This takes the general form:
        p = 2J, q = p - 1
    """
        
    if p==1:
        warnings.warn("CARMA terms are p = 1, q = 0, use fit_drw instead.")
    
    # Sort data
    ind = np.argsort(x)
    x = x[ind]; y = y[ind]; yerr = yerr[ind]
    baseline = x[-1]-x[0]
    
    # Use uniform prior with default 'smart' bounds:
    if bounds == 'default':
        min_precision = np.min(yerr.value)
        amplitude = np.max(y.value)-np.min(y.value)
        amin = np.log(0.1*min_precision)
        amax = np.log(10*amplitude)
        log_a = np.mean([amin,amax])
        
        min_cadence = np.min(np.diff(x.value))
        cmin = np.log(1/(10*baseline.value))
        cmax = np.log(1/min_cadence)
        log_c = np.mean([cmin,cmax])
    else:
        amin = -np.inf
        amax = np.inf
        cmin = -np.inf
        cmax = np.inf
        log_a = 0
        log_c= 0
    
    # Add CARMA parts
    kernel = terms.ComplexTerm(log_a=log_a, log_b=log_a, log_c=log_c, log_d=log_c,
                               bounds=dict(log_a=(amin, amax),
                                           log_c=(cmin, cmax)))
    for j in range(2, p+1):
        kernel += terms.ComplexTerm(log_a=log_a, log_b=log_a, log_c=log_c, log_d=log_c,
                               bounds=dict(log_a=(amin, amax), log_b=(amin, amax),
                                           log_c=(cmin, cmax), log_d=(cmin, cmax)))
    
    # Add jitter term
    if jitter:
        kernel += terms.JitterTerm(log_sigma=1.0, bounds=dict(log_sigma=(0.1*amin, amax)))
       
    gp, samples = fit_celerite(x, y, yerr, kernel, init=init, nburn=nburn, nsamp=nsamp, color=color, plot=plot, verbose=verbose)
        
    # Return the GP model and sample chains
    return gp, samples
        
def fit_celerite(x, y, yerr, kernel, init="minimize", nburn=500, nsamp=2000, color="#ff7f0e", plot=True, verbose=True):
    """
    Fit model to data using a given celerite kernel. Computes the PSD and generates useful plots.
    
    Note: x, y, and yerr must by astropy Quantities with units!
    x: time [astropy unit quantity]
    y: data [astropy unit quantity]
    yerr: error on data [astropy unit quantity]
    kernel: celerite kernel
    init: 'minimize', 'differential_evolution', or array of user-specified (e.g. previous) initial conditions
    nburn: number of burn-in samples
    nsamp: number of production samples
    color: color for plotting
    plot: whether to plot the result
    verbose: whether to print useful messages
    """
    
    baseline = x[-1]-x[0]
    
    gp = celerite.GP(kernel, mean=np.mean(y.value), fit_mean=False)
    gp.compute(x.value, yerr.value)
    if verbose:
        print("Initial log-likelihood: {0}".format(gp.log_likelihood(y.value)))

    # Define a cost function
    def neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y)

    def grad_neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.grad_log_likelihood(y)[1]

    # Fit for the maximum likelihood parameters
    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    
    # MLE solution
    if init == "minimize":
        soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                        method="L-BFGS-B", bounds=bounds, args=(y.value, gp))
        initial = np.array(soln.x)
        if verbose:
            print("Final log-likelihood: {0}".format(-soln.fun))
    elif init == "differential_evolution":
        soln = differential_evolution(neg_log_like, bounds=bounds, args=(y.value, gp))
        initial = np.array(soln.x)
        if verbose:
            print("Final log-likelihood: {0}".format(-soln.fun))
    # Use user-provided initial MLE conditions
    elif np.issubdtype(np.array(init).dtype, np.number):
        initial = init
    else:
        raise ValueError('initial value not recognized!')
        
    gp.set_parameter_vector(initial)

    # Make the maximum likelihood prediction
    t = np.linspace(np.min(x.value)-100, np.max(x.value)+100, 500)
    mu, var = gp.predict(y.value, t, return_var=True)
    std = np.sqrt(var)
    
    # Define the log probablity
    def log_probability(params):
        gp.set_parameter_vector(params)
        lp = gp.log_prior()
        if not np.isfinite(lp):
            return -np.inf
        return gp.log_likelihood(y) + lp

    ndim, nwalkers = len(initial), 32
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    
    if verbose:
        print("Running burn-in...")
    p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
    p0, lp, _ = sampler.run_mcmc(p0, nburn)
    
    if verbose:
        print("Running production...")
    sampler.reset()
    sampler.run_mcmc(p0, nsamp);
        
    # Get posterior and uncertianty
    samples = sampler.flatchain
    s = np.median(samples, axis=0)
    gp.set_parameter_vector(s)
    mu, var = gp.predict(y.value, t, return_var=True)
    std = np.sqrt(var)
    # Noise level
    noise_level = 2.0*np.median(np.diff(x.value))*np.mean(yerr.value**2)
    log_tau_drw = np.log10(1/np.exp(samples[:,1]))
    
    # Lomb-Scargle periodogram with PSD normalization
    freqLS, powerLS = LombScargle(x, y, yerr).autopower(normalization='psd')
    powerLS /= len(x)
    f = np.logspace(np.log10(np.min(freqLS.value)), np.log10(np.max(freqLS.value)), 1000)/u.day
    # Binned Lomb-Scargle periodogram
    num_bins = 12
    f_bin = np.logspace(np.log10(np.min(freqLS.value)), np.log10(np.max(freqLS.value)), num_bins+1)
    psd_binned = np.empty((num_bins, 3))
    f_bin_center = np.empty(num_bins)
    for i in range(num_bins):
        if i == 1:
            idx = (freqLS.value < f_bin[i+1])
        else:
            idx = (freqLS.value >= f_bin[i]) & (freqLS.value < f_bin[i+1])
        len_idx = len(freqLS.value[idx])
        if len(idx) < 2:
            continue
        f_bin_center[i] = np.mean(freqLS.value[idx]) # freq center
        meani = np.mean(powerLS.value[idx])
        stdi = meani/np.sqrt(len_idx)
        psd_binned[i, 0] = meani
        psd_binned[i, 1] = meani + stdi # hi
        psd_binned[i, 2] = meani - stdi # lo
    # The posterior PSD
    psd_samples = np.empty((len(f), len(samples)))
    for i, s in enumerate(samples):
        gp.set_parameter_vector(s)
        psd_samples[:, i] = kernel.get_psd(2*np.pi*f.value)/2*np.pi
    # Compute credibility interval
    psd_credint = np.empty((len(f), 3))
    psd_credint[:, 0] = np.percentile(psd_samples, 16, axis=1)
    psd_credint[:, 2] = np.percentile(psd_samples, 84, axis=1)
    psd_credint[:, 1] = np.median(psd_samples, axis=1)
    
    # Plot
    if plot:
        fig, axs = plt.subplots(2,2, figsize=(15,10), gridspec_kw={'width_ratios': [1, 1.5]})
        # PSD
        axs[0,0].set_xlim(np.min(freqLS.value), np.max(freqLS.value))
        axs[0,0].loglog(freqLS, powerLS, c='grey', lw=2, alpha=0.3, label=r'Lomb$-$Scargle', drawstyle='steps-pre')
        axs[0,0].fill_between(f_bin_center[1:], psd_binned[1:, 1], psd_binned[1:, 2], alpha=0.8, interpolate=True, label=r'binned Lomb$-$Scargle', color='k', step='mid')
        axs[0,0].fill_between(f, psd_credint[:, 2], psd_credint[:, 0], alpha=0.3, label='posterior PSD', color=color)
        xlim = axs[0,0].get_xlim()
        axs[0,0].hlines(noise_level, xlim[0], xlim[1], color='grey', lw=2)
        axs[0,0].annotate("Measurement Noise Level", (1.25 * xlim[0], noise_level / 1.9), fontsize=14)
        axs[0,0].set_ylabel("Power (mag$^2 / $day$^{-1}$)",fontsize=18)
        #axs[0,0].set_ylabel("Power ($\mathrm{ppm}^2 / {0.unit:s}^{-1}$)".format(x[0]),fontsize=18)
        #axs[0,0].set_xlabel("Frequency ({0.unit:s}$^{-1}$)".format(x[0]),fontsize=18)
        axs[0,0].set_xlabel("Frequency (days$^{-1}$)",fontsize=18)
        axs[0,0].tick_params('both',labelsize=16)
        axs[0,0].legend(fontsize=16, loc=1)
        axs[0,0].set_ylim(noise_level / 10.0, 10*axs[0,0].get_ylim()[1])

        # Light curve & prediction
        axs[0,1].errorbar(x.value, y.value, yerr=yerr.value, c='k', fmt='.', alpha=0.75, elinewidth=1)
        axs[0,1].fill_between(t, mu+std, mu-std, color=color, alpha=0.3, label='posterior prediction')
        if True:
            axs[0,1].set_xlabel("Time (MJD)",fontsize=18)
        else:
            axs[0,1].set_xlabel("Time ({0.unit:s})".format(x[0]),fontsize=18)
        axs[0,1].set_ylabel(r'Magnitude',fontsize=18)
        axs[0,1].tick_params(labelsize=18)
        axs[0,1].set_ylim(np.max(y.value) + .1, np.min(y.value) - .1)
        axs[0,1].set_xlim(np.min(t), np.max(t))
        axs[0,1].legend(fontsize=16, loc=1)
    
        # tau_DRW posterior distribution
        axs[1,0].hist(log_tau_drw, color=color, alpha=0.8, fill=None, histtype='step', lw=3, bins=50, label=r'posterior distribution')
        ylim = axs[1,0].get_ylim()
        axs[1,0].vlines(np.percentile(log_tau_drw, 16), ylim[0], ylim[1], color='grey', lw=2, linestyle='dashed')
        axs[1,0].vlines(np.percentile(log_tau_drw, 84), ylim[0], ylim[1], color='grey', lw=2, linestyle='dashed')
        axs[1,0].axvspan(np.log10(0.2*baseline.value), np.max(log_tau_drw), alpha=0.2, color='k')
        axs[1,0].set_ylim(ylim)
        axs[1,0].set_xlim(np.min(log_tau_drw), np.max(log_tau_drw))
        axs[1,0].set_xlabel(r'$\log_{10} \tau_{\rm{DRW}}$ (days)',fontsize=18)
        axs[1,0].set_ylabel('Count',fontsize=18)
        axs[1,0].tick_params(labelsize=18)
    
        # ACF of sq. res.
        s = np.median(samples,axis=0)
        gp.set_parameter_vector(s)
        mu, var = gp.predict(y.value, x.value, return_var=False)
        res2 = (y.value-mu)**2

        # Plot ACF
        lags, c, l, b = axs[1,1].acorr(res2-np.mean(res2), maxlags=None, lw=2, color='k')
        maxlag = (len(lags)-2)/2
        # White noise
        wnoise_upper = 1.96/np.sqrt(len(x))
        wnoise_lower = -1.96/np.sqrt(len(x))
        axs[1,1].fill_between([0, maxlag], wnoise_upper, wnoise_lower, facecolor='lightgrey')
        axs[1,1].set_ylabel(r'ACF $\chi^2$',fontsize=18)
        axs[1,1].set_xlabel(r'Time Lag (days)',fontsize=18)
        axs[1,1].set_xlim(0, maxlag)
        axs[1,1].tick_params('both',labelsize=16)

        fig.tight_layout()
        plt.show()
    
    # Return the GP model and sample chains
    return gp, samples
