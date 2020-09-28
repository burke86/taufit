### Estimating the DRW timescale in AGN light curves using [celerite](https://github.com/dfm/celerite)

#### Author: [Colin J. Burke](https://astro.illinois.edu/directory/profile/colinjb2), University of Illinois at Urbana-Champaign

This repository is intended to be a simple resource for modeling AGN light curves using celerite-based gaussian processes. This code is easy to install and ~10x faster than carma_pack. Only a few lines of code are required to fit each light curve. This package is particularly useful for efficiently extracting the DRW timescale parameter τ_DRW:

```
fit_drw(x*u.day, y*u.mag, yerr*u.mag)
```

![example](https://user-images.githubusercontent.com/13906989/92708436-bc7a0000-f31b-11ea-8e72-474e1c2a3bcd.png)

See [demo.ipynb](https://nbviewer.jupyter.org/github/burke86/taufit/blob/master/demo.ipynb) for usage.

### Setup:

Clone this repository, then:

```
pip install -r requirements.txt
```

