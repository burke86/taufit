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


def simulate_drw(x, tau=50, sigma=0.2, ymean=0, size=1, seed=None):
    """
    Simulate DRW given input times, tau, amplitude, and ymean
    
    x: time (rest frame)
    tau: DRW timescale
    sigma: structure function amplitude
    ymean: data mean
    size: number of samples
    seed: seed for numpy's random number generator (use 'None' to re-seed)

        SFinf = sigma * sqrt(tau / 2)
        
    Note: If tau is too small relative to the sampling x, this may return nans.
    Use a finer sampling and interpolate in this case.
        
    returns: y simulated light curve samples of shape [size, len(x)]
    """
    
    np.random.seed(seed)
    
    log_a = np.log(2*sigma**2)
    log_c = np.log(1/tau)
    kernel = terms.RealTerm(log_a=log_a, log_c=log_c)
    
    # Simulate
    gp = celerite.GP(kernel, mean=ymean)
    gp.compute(x)
    
    y = gp.sample(size=size)
    
    return y
    

def hampel_filter(x, y, window_size, n_sigmas=3):
    """
    Perform outlier rejection using a Hampel filter
    
    x: time (list or np array)
    y: value (list or np array)
    window_size: window size to use for Hampel filter
    n_sigmas: number of sigmas to reject outliers past
    
    returns: x, y, mask [lists of cleaned data and outlier mask]
        
    Adapted from Eryk Lewinson
    https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d
    """
    
    # Ensure data are sorted
    if np.all(np.diff(x) > 0):
        ValueError('Data are not sorted!')
        
    x0 = x[0]
    
    n = len(x)
    outlier_mask = np.zeros(n)
    k = 1.4826 # MAD scale factor for Gaussian distribution
    
    # Loop over data points
    for i in range(n):
        # Window mask
        mask = (x > x[i] - window_size) & (x < x[i] + window_size)
        if len(mask) == 0:
            idx.append(i)
            continue
        # Compute median and MAD in window
        y0 = np.median(y[mask])
        S0 = k*np.median(np.abs(y[mask] - y0))
        # MAD rejection
        if (np.abs(y[i] - y0) > n_sigmas*S0):
            outlier_mask[i] = 1
            
    outlier_mask = outlier_mask.astype(np.bool)
    
    return np.array(x)[~outlier_mask], np.array(y)[~outlier_mask], outlier_mask


def fit_drw(x, y, yerr, init='minimize', nburn=500, nsamp=2000, bounds='default', jitter=False, target_name=None, color="#ff7f0e", plot=True, verbose=True, supress_warn=False, seed=None):
    """
    Fit DRW model using celerite
    
    x: time
    y: data
    yerr: error on data
    init: 'minimize', 'differential_evolution', or array of user-specified (e.g. previous) initial conditions
    nburn: number of burn-in samples
    nsamp: number of production samples
    bounds: 'dafault', 'none', or array of user-specified bounds
    jitter: whether to add jitter (white noise term)
    target_name: name of target to display in light curve legend
    color: color for plotting
    plot: whether to plot the result
    verbose: whether to print useful messages
    supress_warn: whether to supress warnings
    seed: seed for random number generator
    
    returns: gp, samples, fig (celerite GuassianProcess object, samples array, and figure [None if plot=False])
    """
        
    # Sort data
    ind = np.argsort(x)
    x = x[ind]; y = y[ind]; yerr = yerr[ind]
    baseline = x[-1]-x[0]
    
    # Check inputs
    assert (len(x) == len(y) == len(yerr)), "Input arrays must be of equal length."
    # Assign units
    if isinstance(x, u.Quantity):
        if not x.unit == u.day:
            x = x.to(u.day)
    else:
        x = x*u.day
        
    if isinstance(y, u.Quantity):
        assert (y.unit == yerr.unit), "y and yerr must have the same units."
        assert y.unit == u.mag or y.unit == u.dimensionless_unscaled, "y and yerr must have mag or dimensionless_unscaled units, or no units (in which case 'normalized flux' units are assumed)."
    else:
        # Normalize the data
        norm = np.median(y)
        y = y/norm*u.dimensionless_unscaled
        yerr = yerr/norm*u.dimensionless_unscaled
    
    # Use uniform prior with default 'smart' bounds:
    if bounds == 'default':
        min_precision = np.min(yerr.value)
        amplitude = np.max(y.value+yerr.value)-np.min(y.value-yerr.value)
        amin = np.log(0.001*min_precision)
        amax = np.log(10*amplitude)
        log_a = np.mean([amin,amax])
        
        min_cadence = np.clip(np.min(np.diff(x.value)), 0.1, None)
        cmin = np.log(1/(10*baseline.value))
        cmax = np.log(1/min_cadence)
        log_c = np.mean([cmin,cmax])
                
        smin = -10
        smax = np.log(amplitude)
        log_s = np.mean([smin,smax])
    # No bounds
    elif bounds == 'none':
        amin = -np.inf
        amax = np.inf
        cmin = -np.inf
        cmax = np.inf
        smin = -np.inf
        smax = np.inf
        log_a = 0
        log_c = 0
        log_s = 0
    # User-defined bounds
    elif np.issubdtype(np.array(bounds).dtype, np.number):
        amin = bounds[0]
        amax = bounds[1]
        cmin = bounds[2]
        cmax = bounds[3]
        log_a = np.mean([amin,amax])
        log_c = np.mean([cmin,cmax])
        if jitter:
            smin = bounds[4]
            smax = bounds[5]
            log_s = np.mean([smin,smax])
    else:
        raise ValueError('bounds value not recognized!')
                
    kernel = terms.RealTerm(log_a=log_a, log_c=log_c,
                            bounds=dict(log_a=(amin, amax), log_c=(cmin, cmax)))
    
    # Add jitter term
    if jitter:
        kernel += terms.JitterTerm(log_sigma=log_s, bounds=dict(log_sigma=(smin, smax)))
        
    gp, samples, fig = fit_celerite(x, y, yerr, kernel, 1, init=init, nburn=nburn, nsamp=nsamp, target_name=target_name, color=color, plot=plot, verbose=verbose, supress_warn=supress_warn, seed=seed)
    
    # Return the GP model and sample chains
    return gp, samples, fig


def fit_carma(x, y, yerr, p=2, init='minimize', nburn=500, nsamp=2000, bounds='default', jitter=False, target_name=None, color="#ff7f0e", plot=True, verbose=True, supress_warn=False, seed=None):
    """
    Fit CARMA-equivilant model using celerite
    
    Note: x, y, and yerr must by astropy Quantities with units!
    x: time
    y: data
    yerr: error on data
    p: AR order of CARMA model (q = p - 1)
    init: 'minimize', 'differential_evolution', or array of user-specified (e.g. previous) initial conditions
    nburn: number of burn-in samples
    nsamp: number of production samples
    bounds: 'dafault' or array of user-specified bounds
    jitter: whether to add jitter (white noise term)
    target_name: name of target to display in light curve legend
    color: color for plotting
    plot: whether to plot the result
    verbose: whether to print useful messages
    supress_warn: whether to supress warnings
    seed: seed for random number generator
    
    This takes the general form:
        p = J, q = p - 1
            
    returns: gp, samples, fig (celerite GuassianProcess object, samples array, and figure [None if plot=False])
    """
        
    if p==1:
        warnings.warn("CARMA terms are p = 1, q = 0, use fit_drw instead.")
    
    # Sort data
    ind = np.argsort(x)
    x = x[ind]; y = y[ind]; yerr = yerr[ind]
    
    # Check inputs
    assert (len(x) == len(y) == len(yerr)), "Input arrays must be of equal length."
    # Assign units
    if isinstance(x, u.Quantity):
        if not x.unit == u.day:
            x = x.to(u.day)
    else:
        x = x*u.day
        
    if isinstance(y, u.Quantity):
        assert (y.unit == yerr.unit), "y and yerr must have the same units."
        assert y.unit == u.mag or y.unit == u.dimensionless_unscaled, "y and yerr must have mag or dimensionless_unscaled units, or no units (in which case 'normalized flux' units are assumed)."
    else:
        # Normalize the data
        norm = np.median(y)
        y = y/norm*u.dimensionless_unscaled
        yerr = yerr/norm*u.dimensionless_unscaled
    
    # Use uniform prior with default 'smart' bounds:
    if bounds == 'default':
        amin = -10
        amax = 10
        bmin = -10
        bmax = 10
        cmin = -10
        cmax = 10
        dmin = -10
        dmax = 10
        smin = -10
        smax = 10
        log_a = 0
        log_b = 0
        log_c = 0
        log_d = 0
        log_s = 0
    # User-defined bounds (assume each term's bounds are the same for now)
    elif np.issubdtype(np.array(bounds).dtype, np.number):
        amin = bounds[0]
        amax = bounds[1]
        bmin = bounds[2]
        bmax = bounds[3]
        cmin = bounds[4]
        cmax = bounds[5]
        dmin = bounds[6]
        dmax = bounds[7]
        
        log_a = np.mean([amin,amax])
        log_b = np.mean([bmin,bmax])
        log_c = np.mean([cmin,cmax])
        log_d = np.mean([dmin,dmax])
        if jitter:
            smin = bounds[4]
            smax = bounds[5]
            log_s = np.mean([smin,smax])
    else:
        raise ValueError('bounds value not recognized!')
    
    # Add CARMA parts
    kernel = terms.ComplexTerm(log_a=log_a, log_b=log_b, log_c=log_c, log_d=log_d,
                                bounds=dict(log_a=(amin, amax), log_b=(bmin, bmax),
                                           log_c=(cmin, cmax), log_d=(dmin, dmax)))
    
    for j in range(2, p+1):
        kernel += terms.ComplexTerm(log_a=log_a, log_b=log_b, log_c=log_c, log_d=log_d,
                               bounds=dict(log_a=(amin, amax), log_b=(bmin, bmax),
                                           log_c=(cmin, cmax), log_d=(dmin, dmax)))
    
    # Add jitter term
    if jitter:
        kernel += terms.JitterTerm(log_sigma=log_s, bounds=dict(log_sigma=(smin, smax)))
       
    gp, samples, fig = fit_celerite(x, y, yerr, kernel, 2, init=init, nburn=nburn, nsamp=nsamp, target_name=None, color=color, plot=plot, verbose=verbose, supress_warn=supress_warn, seed=seed)
        
    # Return the GP model and sample chains
    return gp, samples, fig


def fit_celerite(x, y, yerr, kernel, tau_term=1, init="minimize", nburn=500, nsamp=2000, target_name=None, color="#ff7f0e", plot=True, verbose=True, supress_warn=False, seed=None):
    """
    Fit model to data using a given celerite kernel. Computes the PSD and generates useful plots.
    
    Note: x, y, and yerr must by astropy Quantities with units!
    x: time [astropy unit quantity]
    y: data [astropy unit quantity]
    yerr: error on data [astropy unit quantity]
    kernel: celerite kernel
    tau_term: index of samples to plot timescale (should be 1 for DRW)
    init: 'minimize', 'differential_evolution', or array of user-specified (e.g. previous) initial conditions
    nburn: number of burn-in samples
    nsamp: number of production samples
    target_name: name of target to display in light curve legend
    color: color for plotting
    plot: whether to plot the result
    verbose: whether to print useful messages
    supress_warn: whether to supress warnings
    seed: seed for random number generator
    
    returns: gp, samples, fig (celerite GuassianProcess object, samples array, and figure [None if plot=False])
    """
    
    # Set seed for reproducability
    np.random.seed(seed)
    
    if supress_warn:
        warnings.filterwarnings("ignore")
    
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
    
    if plot:
        fig = plot_celerite(x, y, yerr, gp, samples, tau_term=tau_term, target_name=target_name, color=color)
    else:
        fig = None
    
    return gp, samples, fig


def plot_celerite(x, y, yerr, gp, samples, tau_term=1, target_name=None, color="#ff7f0e"):
    """
    Plot celerite model, PSD, light curve, and auto-correlation figure
    
    Note: x, y, and yerr must by astropy Quantities with units!
    x: time [astropy unit quantity]
    y: data [astropy unit quantity]
    yerr: error on data [astropy unit quantity]
    kernel: celerite kernel
    gp: celerite GuassianProccess object
    samples: celerite samples array
    tau_term: index of samples to plot timescale (should be 1 for DRW)
    target_name: name of target to display in light curve legend
    color: color for plotting
    
    returns: fig (matplotlib Figure object)
    """
    
    baseline = x[-1]-x[0]
    cadence = np.mean(np.diff(x))
    
    s = np.median(samples, axis=0)
    gp.set_parameter_vector(s)
    
    kernel = gp.kernel
    
    pad = 0.05*baseline.value # 5% padding for plot
    t = np.linspace(np.min(x.value) - pad, np.max(x.value) + pad, 500)
    
    mu, var = gp.predict(y.value, t, return_var=True)
    std = np.sqrt(var)
    # Noise level
    #noise_level = 2.0*np.median(np.diff(x.value))*np.mean(yerr.value**2)
    # Should include jitter noise
    
    # Lomb-Scargle periodogram with PSD normalization
    freqLS, powerLS = LombScargle(x, y, yerr).autopower(normalization='psd')
    powerLS /= len(x)
    f = np.logspace(np.log10(np.min(freqLS.value)), np.log10(np.max(freqLS.value)), 1000)/u.day
    """
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
        if len_idx < 2:
            continue
        f_bin_center[i] = np.mean(freqLS.value[idx]) # freq center
        meani = np.mean(powerLS.value[idx])
        stdi = meani/np.sqrt(len_idx)
        psd_binned[i, 0] = meani
        psd_binned[i, 1] = meani + stdi # hi
        psd_binned[i, 2] = meani - stdi # lo
    """
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
    
    # Do the normalization empirically
    f_norm = np.max(powerLS.value)/psd_credint[0, 1]
    
    psd_credint[:, 0] = psd_credint[:, 0]*f_norm
    psd_credint[:, 2] = psd_credint[:, 2]*f_norm
    psd_credint[:, 1] = psd_credint[:, 1]*f_norm
    
    # Plot
    fig, axs = plt.subplots(2,2, figsize=(15,10), gridspec_kw={'width_ratios': [1, 1.5]})
    # PSD
    axs[0,0].loglog(freqLS, powerLS, c='grey', lw=2, alpha=0.3, label=r'Lomb$-$Scargle', drawstyle='steps-pre')
    #axs[0,0].fill_between(f_bin_center[1:], psd_binned[1:, 1], psd_binned[1:, 2], alpha=0.8, interpolate=True, label=r'binned Lomb$-$Scargle', color='k', step='mid')
    axs[0,0].fill_between(f, psd_credint[:, 2], psd_credint[:, 0], alpha=0.3, label='posterior PSD', color=color)
    #xlim = axs[0,0].get_xlim()
    #axs[0,0].hlines(noise_level, xlim[0], xlim[1], color='grey', lw=2)
    #axs[0,0].annotate("Measurement Noise Level", (1.25 * xlim[0], noise_level / 1.9), fontsize=14)
    if y.unit == u.mag:
        axs[0,0].set_ylabel("Power (mag$^2 / $day$^{-1}$)", fontsize=18)
    else:
        axs[0,0].set_ylabel("Power (ppm$^2 / $day$^{-1}$)", fontsize=18)
    axs[0,0].set_xlabel("Frequency (days$^{-1}$)", fontsize=18)
    axs[0,0].tick_params('both', labelsize=16)
    axs[0,0].legend(fontsize=16, loc=1)
    axs[0,0].set_xlim(np.min(freqLS.value), np.max(freqLS.value))
    axs[0,0].set_ylim([np.min(psd_credint[:, 1]), 10*np.max(psd_credint[:, 1])])

    # Light curve & prediction
    axs[0,1].errorbar(x.value, y.value, yerr=yerr.value, c='k', fmt='.', alpha=0.75, elinewidth=1, label=target_name)
    axs[0,1].fill_between(t, mu+std, mu-std, color=color, alpha=0.3, label='posterior prediction')

    axs[0,1].set_xlabel("Time (days)", fontsize=18)
    if y.unit == u.mag:
        axs[0,1].set_ylabel('Magnitude', fontsize=18)
        axs[0,1].invert_yaxis()
    else:
        axs[0,1].set_ylabel('Normalized Flux', fontsize=18)
    axs[0,1].tick_params(labelsize=18)
    axs[0,1].set_xlim(np.min(t), np.max(t))
    axs[0,1].legend(fontsize=16, loc=1)

    # Plot timescale posterior
    tau = 1/np.exp(samples[:,tau_term])
    log_tau = np.log10(tau)
    tau_med = np.median(tau)
    tau_err_lo = tau_med - np.percentile(tau, 16)
    tau_err_hi = np.percentile(tau, 84) - tau_med
    
    if tau_term  == 1:
        # log tau DRW
        axs[1,0].set_xlabel(r'$\log_{10} \tau_{\rm{DRW}}$ (days)', fontsize=18)
        text = r"$\tau_{\rm{DRW}}={%.1f}^{+%.1f}_{-%.1f}$" % (tau_med, tau_err_hi, tau_err_lo)
    else: # CARMA
        # Plot first order timescale term
        axs[1,0].set_xlabel(r'$\log_{10} 1/c_1$ (days)', fontsize=18)
        text = r"$\log_{10} 1/c_1$ (days)={%.1f}^{+%.1f}_{-%.1f}$" % (tau_med, tau_err_hi, tau_err_lo)
    axs[1,0].text(0.5, 0.9, text, transform=axs[1,0].transAxes, ha='center', fontsize=16)

    # tau_DRW posterior distribution
    axs[1,0].hist(log_tau[np.isfinite(log_tau)], color=color, alpha=0.8, fill=None, histtype='step', lw=3, bins=50, label=r'posterior distribution')
    ylim = axs[1,0].get_ylim()
    axs[1,0].vlines(np.percentile(log_tau, 16), ylim[0], ylim[1], color='grey', lw=2, linestyle='dashed')
    axs[1,0].vlines(np.percentile(log_tau, 84), ylim[0], ylim[1], color='grey', lw=2, linestyle='dashed')
    axs[1,0].axvspan(np.log10(0.2*baseline.value), np.max(log_tau), alpha=0.2, color='k')
    axs[1,0].axvspan(np.min(log_tau), np.log10(cadence.value), alpha=0.2, color='k')
    axs[1,0].set_ylim(ylim)
    axs[1,0].set_xlim(np.min(log_tau[np.isfinite(log_tau)]), np.max(log_tau[np.isfinite(log_tau)]))
    axs[1,0].set_ylabel('Count', fontsize=18)
    axs[1,0].tick_params(labelsize=18)

    # ACF of sq. res.
    s = np.median(samples,axis=0)
    gp.set_parameter_vector(s)
    mu, var = gp.predict(y.value, x.value, return_var=False)
    res2 = (y.value - mu)**2

    # Plot ACF
    lags, c, l, b = axs[1,1].acorr(res2 - np.mean(res2), maxlags=None, lw=2, color='k')
    maxlag = (len(lags)-2)/2
    # White noise
    wnoise_upper = 1.96/np.sqrt(len(x))
    wnoise_lower = -1.96/np.sqrt(len(x))
    axs[1,1].fill_between([0, maxlag], wnoise_upper, wnoise_lower, facecolor='lightgrey')
    axs[1,1].set_ylabel(r'ACF $\chi^2$', fontsize=18)
    axs[1,1].set_xlabel(r'Time Lag (days)', fontsize=18)
    axs[1,1].set_xlim(0, maxlag)
    axs[1,1].tick_params('both', labelsize=16)

    fig.tight_layout()
    plt.show()
    
    # Return the figure
    return fig
