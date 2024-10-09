import os, sys
import warnings
import celerite
from celerite import terms
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def smoothly_broken_power_law(f, A=1, f_br=1e-3, alpha=0, beta=2):
    return A/((f/f_br)**alpha + (f/f_br)**beta)


def simulate_from_psd(S_func, m=2000, dt=1, ymean=0, sigma=0.2, size=1, seed=None, **args):
    """
    Simulate light curve given input times, model PSD, and ymean
    
    S_func: model PSD function S(omega) [note omega = 2 pi f]
    m: number of bins [output will have length 2(m - 1)]
    dt: equal spacing in time
    ymean: data mean
    sigma: data standard deviation
    size: number of samples
    seed: seed for numpy's random number generator (use 'None' to re-seed)
        
    returns:
    x:
    y: simulated light curve samples of shape [size, 2(m - 1)]
    omega: 
    
    Adapted from Timmer, J. & Koenig, M. On Generating Power Law Noise. A&A 300 707 (1995)
    and https://github.com/pabell/pylcsim/blob/master/lcpsd.py
    """
    
    np.random.seed(seed)
    
    n = 2*(m - 1)
    
    # Get FFT frequencies
    f = np.fft.rfftfreq(n=n, d=dt)[1:]
    omega = f*2*np.pi
    
    # Evaluate PSD function
    S = S_func(omega, **args)
    
    # Model PSD factor
    fac = np.sqrt(S/2.0)
        
    y = np.zeros([size, n-2])
    
    for i in range(size):
    
        # Generate the real and imaginary terms
        re = np.random.normal(size=n//2)*fac
        im = np.random.normal(size=n//2)*fac
        
        # Generate randomized PSD
        S_rand = re + 1j*im

        yi = np.fft.irfft(S_rand)
        
        # Renormalize the light curve
        mean = np.mean(yi)
        std = np.std(yi)
        y[i,:] = (yi - mean)/std*sigma + ymean

    # Times
    x = dt*np.arange(0, n-2)
    
    return x, y, f, S


def fit_drw(x, y, yerr, init='minimize', nburn=500, nsamp=2000, lamb=None, bounds='default', target_name=None, color="#ff7f0e", plot=True, verbose=True, supress_warn=False, seed=None):
    """
    Fit DRW model using celerite
    
    x: time
    y: data
    yerr: error on data
    init: 'minimize', 'differential_evolution', or array of user-specified (e.g. previous) initial conditions
    nburn: number of burn-in samples
    nsamp: number of production samples
    bounds: 'dafault', 'none', or array of user-specified bounds
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
        
        min_cadence = np.clip(np.min(np.diff(x.value)), 1e-8, None)
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
    kernel += terms.JitterTerm(log_sigma=log_s, bounds=dict(log_sigma=(smin, smax)))
        
    gp, samples, fig = fit_celerite(x, y, yerr, kernel, init=init, nburn=nburn, nsamp=nsamp, lamb=lamb, target_name=target_name, color=color, plot=plot, verbose=verbose, supress_warn=supress_warn, seed=seed)
    
    # Return the GP model and sample chains
    return gp, samples, fig


def fit_carma(x, y, yerr, p=2, init='minimize', nburn=500, nsamp=2000, lamb=None, bounds='default', target_name=None, color="#ff7f0e", plot=True, verbose=True, supress_warn=False, seed=None):
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
    kernel += terms.JitterTerm(log_sigma=log_s, bounds=dict(log_sigma=(smin, smax)))
       
    gp, samples, fig = fit_celerite(x, y, yerr, kernel, init=init, nburn=nburn, nsamp=nsamp, lamb=lamb, target_name=None, color=color, plot=plot, verbose=verbose, supress_warn=supress_warn, seed=seed)
        
    # Return the GP model and sample chains
    return gp, samples, fig


def fit_celerite(x, y, yerr, kernel, init="minimize", nburn=500, nsamp=2000, lamb=None, target_name=None, color="#ff7f0e", plot=True, verbose=True, supress_warn=False, seed=None):
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
    
    def _mle(y, gp, initial_params, bounds):
    
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
            
        return initial
    
    # Find MLE
    initial = _mle(y, gp, initial_params, bounds)   
    gp.set_parameter_vector(initial)
    
    # Filter long-term trends
    if lamb is not None:
        import statsmodels.api as sm
        # Filter on evenly-sampled MLE solution
        mean_cadence = np.mean(np.diff(x.value))
        t = np.arange(np.min(x.value), np.max(x.value), mean_cadence/10)
        mu = gp.predict(y.value, t, return_cov=False)
        cycle, trend = sm.tsa.filters.hpfilter(mu, lamb)
        
        if plot:
            # Light curve & prediction
            fig, ax_lc = plt.subplots(1,1, figsize=(9,5))
            ax_lc.errorbar(x.value, y.value, yerr=yerr.value, c='k', fmt='.', alpha=0.75, elinewidth=1, label=target_name)
            ax_lc.plot(t, trend, c='r', label='trend')
            ax_lc.invert_yaxis()
            ax_lc.set_xlabel("Time (days)", fontsize=20)
            if y.unit == u.mag:
                ax_lc.set_ylabel("Magnitude", fontsize=20)
            else:
                ax_lc.set_ylabel("Normalized Flux", fontsize=20)

            ax_lc.minorticks_on()
            ax_lc.tick_params('both',labelsize=18)
            ax_lc.tick_params(axis='both', which='both', direction='in')
            ax_lc.tick_params(axis='both', which='major', length=6)
            ax_lc.tick_params(axis='both', which='minor', length=3)
            ax_lc.xaxis.set_ticks_position('both')
            ax_lc.yaxis.set_ticks_position('both')

            ax_lc.legend(fontsize=16, loc=1)
            fig.tight_layout()

        # Subtract trend at real data
        y = y - np.interp(x.value, t, trend)*(y.unit) + np.median(y)
        
        # Find new MLE
        gp = celerite.GP(kernel, mean=np.mean(y.value), fit_mean=False)
        gp.compute(x.value, yerr.value)
        # Fit for the maximum likelihood parameters
        initial_params = gp.get_parameter_vector()
        bounds = gp.get_parameter_bounds()
    
        initial = _mle(y, gp, initial_params, bounds)   
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
        fig = plot_celerite(x, y, yerr, gp, samples, target_name=target_name, color=color)
    else:
        fig = None
    
    return gp, samples, fig


def plot_celerite(x, y, yerr, gp, samples, target_name=None, color="#ff7f0e"):
    """
    Plot celerite model, PSD, light curve, and auto-correlation figure
    
    Note: x, y, and yerr must by astropy Quantities with units!
    x: time [astropy unit quantity]
    y: data [astropy unit quantity]
    yerr: error on data [astropy unit quantity]
    gp: celerite GuassianProccess object
    samples: celerite samples array
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
    
    fig_corner, axs = plt.subplots(3,3, figsize=(5,5))
    
    # Corner plot
    samples_sf = [np.log10(np.sqrt(np.exp(samples[:,0]/2))), np.log10(1/np.exp(samples[:,1])), np.log10(np.exp(samples[:,2]))]
    samples_sf = np.array(samples_sf).T
    fig_corner = corner.corner(samples_sf, show_titles=True, fig=fig_corner, quantiles=[0.16,0.84],
    labels = [r"$\log_{10}\ \sigma_{\rm{DRW}}$", r"$\log_{10}\ \tau_{\rm{DRW}}$", r"$\log_{10}\ \sigma_{\rm{n}}$"],
    label_kwargs = dict(fontsize=18), title_kwargs=dict(fontsize=10));
    # Ticks in
    axs = np.array(fig_corner.axes).reshape((3, 3))
    for i in range(3):
        for j in range(3):
            ax = axs[i,j]
            ax.tick_params('both',labelsize=12)
            ax.tick_params(axis='both', which='both', direction='in')
            ax.tick_params(axis='both', which='major', length=6)
            ax.tick_params(axis='both', which='minor', length=3)

    # Draw bad tau region
    axs[1,1].axvspan(np.log10(0.2*baseline.value), np.log10(10*baseline.value), color= "red", zorder=-5, alpha=0.2)
    axs[2,1].axvspan(np.log10(0.2*baseline.value), np.log10(10*baseline.value), color= "red", zorder=-5, alpha=0.2)
    
    # Add gridspec to corner plot and re-arrange later (pretty hacky, but it works)
    gs = gridspec.GridSpec(ncols=4, nrows=4, figure=fig_corner)

    # Light curve plot
    ax_lc = fig_corner.add_subplot(gs[0, :])
    box = ax_lc.get_position()
    box.x0 = box.x0 + 0.2
    box.x1 = box.x1 + 1.0
    box.y0 = box.y0 + 0.4
    box.y1 = box.y1 + 0.9
    ax_lc.set_position(box)
    
    # Light curve & prediction
    ax_lc.errorbar(x.value, y.value, yerr=yerr.value, c='k', fmt='.', alpha=0.75, elinewidth=1, label=target_name)
    ax_lc.fill_between(t, mu+std, mu-std, color="#ff7f0e", alpha=0.3, label='DRW prediction')
    ax_lc.set_xlim(np.min(t), np.max(t))
    ax_lc.invert_yaxis()
    ax_lc.set_xlabel("Time (days)", fontsize=20)
    if y.unit == u.mag:
        ax_lc.set_ylabel("Magnitude", fontsize=20)
    else:
        ax_lc.set_ylabel("Normalized Flux", fontsize=20)

    ax_lc.minorticks_on()
    ax_lc.tick_params('both',labelsize=18)
    ax_lc.tick_params(axis='both', which='both', direction='in')
    ax_lc.tick_params(axis='both', which='major', length=6)
    ax_lc.tick_params(axis='both', which='minor', length=3)
    ax_lc.xaxis.set_ticks_position('both')
    ax_lc.yaxis.set_ticks_position('both')
    
    ax_lc.legend(fontsize=16, loc=1)
    
    # PSD Plot
    ax_psd = fig_corner.add_subplot(gs[:, -1])
    fig_corner.set_size_inches([6,6])
    # Move the subplot over
    box = ax_psd.get_position()
    box.x0 = box.x0 + 0.4
    box.x1 = box.x1 + 1.2
    ax_psd.set_position(box)
    
    # Lomb-Scargle periodogram with PSD normalization
    freqLS, powerLS = LombScargle(x, y, yerr).autopower(normalization='psd')
    #powerLS /= len(x) # Celerite units
    fs = (1./(np.min(np.diff(x)[np.diff(x)>0])))
    powerLS *= 2/(len(x)*fs) # lightkurve units [flux variance / frequency unit]
    ax_psd.loglog(freqLS.value, powerLS.value, c='grey', lw=1, alpha=0.3, label=r'PSD', drawstyle='steps-pre')
    
    # Celerite posterior PSD
    f_eval = np.logspace(np.log10(freqLS.value[0]), np.log10(freqLS.value[-1]), 150)
    psd_samples = np.empty((len(f_eval), len(samples)))
    for i, s in enumerate(samples):
        gp.set_parameter_vector(s)
        psd_samples[:, i] = kernel.get_psd(2*np.pi*f_eval)/(2*np.pi)
    # Compute credibility interval
    psd_credint = np.empty((len(f_eval), 3))
    psd_credint[:, 0] = np.percentile(psd_samples, 16, axis=1)
    psd_credint[:, 2] = np.percentile(psd_samples, 84, axis=1)
    psd_credint[:, 1] = np.median(psd_samples, axis=1)
    
    # Do the normalization empirically
    f_norm = np.max(powerLS.value[freqLS.value>1/(2*np.pi*0.2*baseline.value)])/psd_credint[0, 1]
    psd_credint[:, 0] = psd_credint[:, 0]*f_norm
    psd_credint[:, 2] = psd_credint[:, 2]*f_norm
    psd_credint[:, 1] = psd_credint[:, 1]*f_norm
    
    ax_psd.fill_between(f_eval, psd_credint[:, 2], psd_credint[:, 0], alpha=0.3, label='Model PSD', color=color)
    
    # Danger zone
    ax_psd.axvspan(np.min(freqLS.value), 1/(2*np.pi*0.2*baseline.value), color='red', zorder=-1, alpha=0.2)
    fmax = (1./(2*np.pi*np.mean(np.diff(x)[np.diff(x)>0])))
    ax_psd.axvspan(fmax.value, np.max(freqLS.value), color='red', zorder=-1, alpha=0.2)
    
    ax_psd.set_xlim([np.min(freqLS.value), np.max(freqLS.value)])
    ax_psd.set_ylim([0.5*np.nanmin(psd_credint[:, 0]), 10*np.nanmax(psd_credint[:, 2])])
    
    ax_psd.minorticks_on()
    ax_psd.tick_params('both',labelsize=18)
    ax_psd.tick_params(axis='both', which='both', direction='in')
    ax_psd.tick_params(axis='both', which='major', length=6)
    ax_psd.tick_params(axis='both', which='minor', length=3)
    ax_psd.xaxis.set_ticks_position('both')
    ax_psd.yaxis.set_ticks_position('both')
    
    ax_psd.legend(fontsize=16, loc=1)
    
    ax_psd.set_xlabel(r'Frequency (days$^{-1}$)', fontsize=20)
    ax_psd.set_ylabel(r'Power [(rms)$^2$ days]', fontsize=20)

    fig_timing = plt.gcf()
    
    # Return the figure (use bbox_inches='tight' when saving)
    return fig_timing


def plot_acf_res(x, y, gp, samples, plot=True):
    """
    Compute/plot ACF of Chi^2 residuals to test model
    
    Note: x, y, and yerr must by astropy Quantities with units!
    x: time [astropy unit quantity]
    y: data [astropy unit quantity]
    gp: celerite GuassianProccess object
    samples: celerite samples array
    plot: whether or not to plot ACF(Chi^2) [bool]
    
    returns: Ljung-Box test p-value at maxlag
    """
    
    import statsmodels.api as sm
    
    s = np.median(samples, axis=0)
    gp.set_parameter_vector(s)
    
    kernel = gp.kernel
    
    mu, var = gp.predict(y.value, x.value, return_var=False)
    
    res2 = (mu - y.value)**2
        
    # Plot auto-correlation function (ACF) of chi^2 residuals
    acf, ci, qstat, pvals = sm.tsa.stattools.acf(res2 - np.mean(res2), qstat=True, alpha=0.05)
    
    if plot:
        sm.graphics.tsa.plot_acf(res2 - np.mean(res2))
    
    """
    lags, c, l, b = axs[1,1].acorr(res2 - np.mean(res2), maxlags=None, lw=2, color='k')
    maxlag = (len(lags)-2)/2
    # White noise
    wnoise_upper = 1.96/np.sqrt(len(x))
    wnoise_lower = -1.96/np.sqrt(len(x))
    ax.fill_between([0, maxlag], wnoise_upper, wnoise_lower, facecolor='lightgrey')
    ax.set_ylabel(r'ACF $\chi^2$', fontsize=18)
    ax.set_xlabel(r'Time Lag (days)', fontsize=18)
    ax.set_xlim(0, maxlag)
    ax.tick_params('both', labelsize=16)
    """

    return pvals[-1]

def p2sigma(p):
    """
    Helper function to convert p-value to sigma (Z-score)
    
    p: p-value
    
    returns: sigma
    """
    
    import scipy.stats as st
    
    # https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa
    # Statisticians call sigma Z-score
    log_p = np.log(p)
    if (log_p > -36):
        sigma = st.norm.ppf(1 - p/2)
    else:
        sigma = np.sqrt(np.log(2/np.pi) - 2*np.log(8.2) - 2*log_p)
    return sigma
