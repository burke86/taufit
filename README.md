### Estimating the DRW timescale in AGN light curves using [celerite](https://github.com/dfm/celerite)

#### Author: [Colin J. Burke](https://astro.illinois.edu/directory/profile/colinjb2), University of Illinois at Urbana-Champaign

This repository is intended to be a simple resource for modeling AGN light curves using celerite-based gaussian processes. This code is easy to install and ~10x faster than carma_pack. Only a few lines of code are required to fit each light curve. This package is particularly useful for efficiently extracting the DRW timescale parameter τ_DRW:

```
fit_drw(x*u.day, y*u.mag, yerr*u.mag)
```

![example](https://user-images.githubusercontent.com/13906989/92708436-bc7a0000-f31b-11ea-8e72-474e1c2a3bcd.png)

See [demo.ipynb](https://nbviewer.jupyter.org/github/burke86/taufit/blob/master/demo.ipynb) for usage.

## ⚠️ Warning

This repository is no longer actively maintained.  
Please consider using **[EzTaoX](https://github.com/LSST-AGN-Variability/EzTaoX)** instead.


### Citation:

If you make use of this code, please cite

```
@ARTICLE{Burke2021,
       author = {{Burke}, Colin J. and {Shen}, Yue and {Blaes}, Omer and {Gammie}, Charles F. and {Horne}, Keith and {Jiang}, Yan-Fei and {Liu}, Xin and {McHardy}, Ian M. and {Morgan}, Christopher W. and {Scaringi}, Simone and {Yang}, Qian},
        title = "{A characteristic optical variability time scale in astrophysical accretion disks}",
      journal = {Science},
     keywords = {ASTRONOMY, Astrophysics - Astrophysics of Galaxies, Astrophysics - High Energy Astrophysical Phenomena},
         year = 2021,
        month = aug,
       volume = {373},
       number = {6556},
        pages = {789-792},
          doi = {10.1126/science.abg9933},
archivePrefix = {arXiv},
       eprint = {2108.05389},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021Sci...373..789B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


### Setup:

Clone this repository, then:

```
pip install -r requirements.txt
```

