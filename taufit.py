import os, sys
import celerite
from celerite import terms
import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.timeseries import LombScargle
from scipy.optimize import minimize

def simulate_drw(t_rest, tau=50, z=0.0, xmean=0, SFinf=0.3):
    # Code adapted from astroML
    #  Xmean = b * tau
    #  SFinf = sigma * sqrt(tau / 2)

    N = len(t_rest)

    t_obs = t_rest*(1 + z)/tau
    np.random.seed()
    x = np.zeros(N)
    x[0] = np.random.normal(xmean, SFinf)
    E = np.random.normal(0, 1, N)

    for i in range(1, N):
        dt = t_obs[i] - t_obs[i - 1]
        x[i] = (x[i - 1]
                - dt * (x[i - 1] - xmean)
                + np.sqrt(2) * SFinf * E[i] * np.sqrt(dt))
        
    return x
    
def fit_drw(x, y, yerr, nburn=500, nsamp=2000, color="#ff7f0e", plot=True, verbose=True):
        
    # Sort data
    ind = np.argsort(x)
    x = x[ind]; y = y[ind]; yerr = yerr[ind]
    
    # Model priors
    min_precision = np.min(yerr.value)
    amplitude = np.max(y.value)-np.min(y.value)
    amin = np.log(0.1*min_precision)
    amax = np.log(10*amplitude)
    
    baseline = x[-1]-x[0]
    min_cadence = np.min(np.diff(x.value))
    cmin = np.log(1/(10*baseline.value))
    cmax = np.log(1/min_cadence)
    
    bounds_drw = dict(log_a=(-15.0, 5.0), log_c=(cmin, cmax))
    kernel = terms.RealTerm(log_a=0, log_c=np.mean([cmin,cmax]), bounds=bounds_drw)
    
    # Jitter?
    #bounds_jitter = dict(log_sigma=(-25.0, 10.0))
    #kernel_jit = terms.JitterTerm(log_sigma=1.0, bounds=bounds_jitter)
    
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
    soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                    method="L-BFGS-B", bounds=bounds, args=(y.value, gp))
    gp.set_parameter_vector(soln.x)
    if verbose:
        print("Final log-likelihood: {0}".format(-soln.fun))

    # Make the maximum likelihood prediction
    t = np.linspace(np.min(x.value)-100, np.max(x.value)+100, 500)
    mu, var = gp.predict(y.value, t, return_var=True)
    std = np.sqrt(var)
    
    # Define the log probablity
    def log_probability(params):
        gp.set_parameter_vector(params)
        lp = gp.log_prior()
        # tau prior
        lp_c = params[1] # log 1/tau
        if not np.isfinite(lp):
            return -np.inf
        return gp.log_likelihood(y) + lp + lp_c

    initial = np.array(soln.x)
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
    num_bins = 12 #len(freqLS)//100
    f_bin = np.logspace(np.log10(np.min(freqLS.value)), np.log10(np.max(freqLS.value)), num_bins+1)
    psd_binned = np.empty((num_bins, 3))
    f_bin_center = np.empty(num_bins)
    for i in range(num_bins):
        idx = (freqLS.value >= f_bin[i]) & (freqLS.value < f_bin[i+1])
        len_idx = len(freqLS.value[idx])
        f_bin_center[i] = np.mean(freqLS.value[idx]) # freq center
        meani = np.mean(powerLS.value[idx])
        stdi = meani/np.sqrt(len_idx)
        psd_binned[i, 0] = meani
        psd_binned[i, 1] = meani + stdi # hi
        psd_binned[i, 2] = meani - stdi # lo
        # If below noise, level break
        if meani < noise_level:
            psd_binned = psd_binned[:i+1, :]
            f_bin_center = f_bin_center[:i+1]
            break
    # The posterior PSD
    #print(psd_binned[1:, 1])
    #print(psd_binned[1:, 2])
    psd_credint = np.empty((len(f), 3))
    # Compute the PSD and credibility interval at each frequency fi
    # Code adapted from B. C. Kelly carma_pack
    for i, fi in enumerate(f.value):
        omegai = 2*np.pi*fi # Convert to angular frequencies
        ai = np.exp(samples[:,0])
        ci = np.exp(samples[:,1])
        psd_samples = np.sqrt(2/np.pi)*ai/ci*(1 + (omegai/ci)**2)**-1
        # Compute credibility interval
        psd_credint[i, 0] = np.percentile(psd_samples, 16, axis=0)
        psd_credint[i, 2] = np.percentile(psd_samples, 84, axis=0)
        psd_credint[i, 1] = np.median(psd_samples, axis=0)
    # Plot
    if plot:
        fig, axs = plt.subplots(1,2, figsize=(15,5), gridspec_kw={'width_ratios': [1, 1.5]})
        axs[0].set_xlim(np.min(freqLS.value), np.max(freqLS.value))
        axs[0].loglog(freqLS, powerLS, c='grey', lw=2, alpha=0.3, label=r'Lomb$-$Scargle', drawstyle='steps-pre')
        axs[0].fill_between(f_bin_center[1:], psd_binned[1:, 1], psd_binned[1:, 2], alpha=0.8, interpolate=True, label=r'binned Lomb$-$Scargle', color='k', step='mid')
        axs[0].fill_between(f, psd_credint[:, 2], psd_credint[:, 0], alpha=0.3, label='posterior PSD', color=color)
        xlim = axs[0].get_xlim()
        axs[0].hlines(noise_level, xlim[0], xlim[1], color='grey', lw=2)
        axs[0].annotate("Measurement Noise Level", (1.25 * xlim[0], noise_level / 1.9), fontsize=14)
        axs[0].set_ylabel("Power ($\mathrm{ppm}^2 / day^{-1}$)",fontsize=18)
        axs[0].set_xlabel("Frequency (days$^{-1}$)",fontsize=18)
        axs[0].tick_params('both',labelsize=16)
        axs[0].legend(fontsize=16, loc=1)
        axs[0].set_ylim(noise_level / 10.0, 10*axs[0].get_ylim()[1])

        # Plot light curve prediction
        axs[1].errorbar(x.value, y.value, yerr=yerr.value, c='k', fmt='.', alpha=0.75, elinewidth=1)
        axs[1].fill_between(t, mu+std, mu-std, color=color, alpha=0.3, label='posterior prediction')
        axs[1].set_xlabel('Time (MJD)',fontsize=18)
        axs[1].set_ylabel(r'Magnitude $g$',fontsize=18)
        axs[1].tick_params(labelsize=18)
        axs[1].set_ylim(np.max(y.value) + .1, np.min(y.value) - .1)
        axs[1].set_xlim(np.min(t), np.max(t))
        axs[1].legend(fontsize=16, loc=1)
        fig.tight_layout()

        fig, axs = plt.subplots(1,2, figsize=(15,5), gridspec_kw={'width_ratios': [1, 1.5]})
    
        axs[0].hist(log_tau_drw, color=color, alpha=0.8, fill=None, histtype='step', lw=3, density=True, bins=50, label=r'posterior distribution')
        ylim = axs[0].get_ylim()
        axs[0].vlines(np.log10(0.2*baseline.value), ylim[0], ylim[1], color='grey', lw=4)
        axs[0].set_ylim(ylim)
        axs[0].set_xlim(np.min(log_tau_drw), np.max(log_tau_drw))
        axs[0].set_xlabel(r'$\log_{10} \tau_{\rm{DRW}}$',fontsize=18)
        axs[0].set_ylabel('count',fontsize=18)
        axs[0].tick_params(labelsize=18)
    
        # ACF of sq. res.
        s = np.median(samples,axis=0)
        gp.set_parameter_vector(s)
        mu, var = gp.predict(y.value, x.value, return_var=False)
        res2 = (y.value-mu)**2

        maxlag = 50

        # Plot ACF
        axs[1].acorr(res2-np.mean(res2), maxlags=maxlag, lw=2)
        # White noise
        wnoise_upper = 1.96/np.sqrt(len(x))
        wnoise_lower = -1.96/np.sqrt(len(x))
        axs[1].fill_between([0, maxlag], wnoise_upper, wnoise_lower, facecolor='grey')
        axs[1].set_ylabel(r'ACF $\chi^2$',fontsize=18)
        axs[1].set_xlabel(r'Time Lag [days]',fontsize=18)
        axs[1].set_xlim(0, maxlag)
        axs[1].tick_params('both',labelsize=16)

        fig.tight_layout()
        plt.show()
    
        # Make corner plot
        # These are natural logs
        #fig = corner.corner(samples, quantiles=[0.16,0.84], show_titles=True,
        #            labels=[r"$\ln\ a$", r"$\ln\ c$"], titlesize=16);
        #for ax in fig.axes:
        #    ax.tick_params('both',labelsize=16)
        #    ax.xaxis.label.set_size(16)
        #    ax.yaxis.label.set_size(16)

        #fig.tight_layout()
        #plt.show()
    
    # Return the GP model and sample chains
    return gp, samples

def asses_bias_drw(x, y, yerr):
    # Asses bias in the fit from sampling effects using simulations with varying tau_drw
    # This usually is not an issue if tau_DRW < 20 % of the baseline light curve
    # Can we recover
    pass
    
def fit_shot(x, y, yerr, nburn=500, nsamp=2000, color="#ff7f0e", target_name=None):
    pass