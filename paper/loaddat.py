import sys, os, glob
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.coordinates import concatenate
plt.rcParams['axes.linewidth'] = 1.5
from astropy.io import fits

import matplotlib

from astropy.utils.data import conf
# The Palomar-Green RAs are incorrect in Kelly et al (2009) (even the erratum is wrong!?) so use the redshift to match those
pg_name = np.genfromtxt('data/pg_coords.txt', unpack=False, dtype=str)
pg_z = np.array([float(i[-1]) for i in pg_name])
pg_name = np.array([' '.join(i[:-1]) for i in pg_name])
pg_ra = np.array([i[:5] for i in pg_name])

# Read Kelly tables
m_name, m_ra, m_dec = np.genfromtxt('data/Magellanic_quasars.dat', unpack=True, usecols=[0,1,2], skip_header=1, dtype=str)
z_kelly, log_L5100_kelly, mass_bh_kelly, mass_bh_err_kelly, ref_kelly = np.genfromtxt('data/apj299016t1_mrt.txt', unpack=True, usecols=[6,7,8,9,10], skip_header=35, dtype=float)
name_kelly = np.genfromtxt('data/apj299016t1_mrt.txt', unpack=False, usecols=[0,1,2,3,4,5], skip_header=35, dtype=str)
name_kelly = np.array([' '.join(i) for i in name_kelly])
#with conf.set_temp('remote_timeout', 90):
coord_kelly = concatenate([SkyCoord.from_name(i) for i in name_kelly])


# Shen SDSS Quasars
hdul_yue = fits.open("data/dr7_bh_May_2011.fits")

# Read Qian's light curves
# Read Qian's database
#hdul_qian = fits.open("data/DB_QSO_S82_all_forColin.fits") # All (only annual cadence for DES)
hdul_qian = fits.open("data/DB_QSO_S82_S12.fits") # Restricted to S1/S2
data_qian = hdul_qian[1].data
dbid_qian = data_qian['DBID']
ra_qian = data_qian['RA']
dec_qian = data_qian['DEC']
# Read Yue's catalog to get virial BH masses
hdul_yue = fits.open("data/dr7_bh_May_2011.fits")
data_yue = hdul_yue[1].data
catalog_yue = SkyCoord(data_yue['RA'], data_yue['DEC'], unit=u.deg)


# Log error = https://faculty.washington.edu/stuve/log_error.pdf
# log10 dy = 0.434 dy / y
def gather_data_kelly(filepath, plot=True, verbose=False):
    
    
    # Load data
    # If starts with number, skip first two lines
    filename = os.path.basename(filepath)
    if filename[0].isdigit():
        skiprows = 2
    else:
        skiprows = 0
    x, y, yerr = np.loadtxt(filepath, unpack=True, usecols=[0,1,2], skiprows=skiprows)
    # Get coordinate names
    # MACHO
    if filename[0].isdigit():
        name = str(m_ra[filename[:-4]==m_name][0])+str(m_dec[filename[:-4]==m_name][0])
        LCref = "Geha et al. (2003)"
        band = 'V'
        unit = 'mag'
        rm = False
        dfname = os.path.basename(filename)[:-4]
    # Palomar-Green
    elif filename[:2] == 'pg':
        # NONE OF THESE ARE ACCEPTABLE
        filename_ra = filename[2:4]+' '+filename[4:6]
        pg_z_i = pg_z[pg_ra==filename_ra][0]
        #name = pg_name[pg_ra==filename_ra][0] wrong!
        name = name_kelly[z_kelly==pg_z_i][0]
        LCref = "Giveon et al. (1999)"
        band = 'R'
        unit = 'mag'
        rm = False # SOME TRUE SOME FALSE
        dfname = 'PG ' + os.path.basename(filename)[2:-4]
    # AGNWatch Name
    else:
        # Generally, these light curves are not very long
        name = filename.split('_')[0]
        LCref = "AGNWatch"
        band = 'R'
        unit = 'mag'
        rm = True # I think??
        if "ngc7496" in name:
            mass = -1 # Skip this galaxy it is in the newer (Peterson 2014 data)
        # clean up names
        if 'ngc' in name:
            dfname = 'NGC ' + name.split('ngc')[1]
        elif 'mrk' in name:
            dfname = 'Mrk ' + name.split('mrk')[1]
        elif 'fairall' in name:
            dfname = 'Fairall 9'
        elif 'akn' in name:
            dfname = 'Akn ' + name.split('akn')[1]
        #
        if '564' in name or 'fairall9' in name or '279' in name or '509' in name or '3783' in name or '4051' in name or '5548' in name:
            unit = r'$10^{-15}$ erg s$^{-1}$ cm$^{-2} \AA^{-1}$'
        elif '4151' in name or '7469' in name:
            unit = r'$10^{-14}$ erg s$^{-1}$ cm$^{-2} \AA^{-1}$'
    # Get coord from name
    coord = SkyCoord.from_name(name)
    # Get properties
    if 'ngc4395_tess' in filename:
        # Woo et al values
        mass = np.log10(9100)
        mass_err = 0.434*1550/9100
        z = 0.001064
        LCref = "Burke et al. (2020)"
        log_Mdot = 0
        log_mdoterr = 0
        log_L5100 = 39.76
        log_L5100err = 0.03
        Lref = 'Cho et al. (2020)'
        massref = 'Woo et al. (2019)'
        rm = True
        band = 'TESS'
        dfname = 'NGC 4395'
        unit = '$e^{-}$ s$^{-1}$'
        # Get properties
    elif 'J0218' in filename:
        # Guo et al (2020) values 
        mass_range = np.array([6.43,6.72])
        mass = np.log10(np.mean(10**mass_range))
        mass_err = 10**mass-10**mass_range[0]
        mass_err = 0.434*mass_err/(10**mass)
        z = 0.823
        ref = "Guo et al. (2020)"
        log_Mdot = 0
        log_mdoterr = 0
        log_L5100 = 43.52
        log_L5100err = 0
        Lref = "Guo et al. (2020)"
        massref = "Guo et al. (2020)"
        LCref = "Guo et al. (2020)"
        band = 'g'
        unit = 'mag'
        dfname = 'DES J0218-0430'
        rm = False
    else:
        # Find closest match to Kelly (2009) table of BH masses
        idx,d2d,d3d = coord.match_to_catalog_sky(coord_kelly)
        mass = mass_bh_kelly[idx]
        mass_err = mass_bh_err_kelly[idx]
        z = z_kelly[idx]
        log_Mdot = 0
        log_mdoterr = 0
        log_L5100 = log_L5100_kelly[idx]
        log_L5100err = 0
        Lref = 'Kelly et al. (2008)' # See refs within
        massref = 'Kelly et al. (2008)'
    if mass == -1:
        return None
    return dfname, coord.ra.deg, coord.dec.deg, mass, mass_err, z, log_L5100, log_L5100err, band, unit, x, y, yerr, rm, LCref, Lref, massref

def gather_data_qian(filepath, plot=True, verbose=False):
    # Load data
    # If starts with number, skip first two lines
    filename = os.path.basename(filepath)
    hdul = fits.open(filepath)
    data_qian = hdul[1].data
    dbid = data_qian['DBID'][0]
    x = data_qian['MJD']
    y = data_qian['MAG']
    yerr = data_qian['MAGERR']
    band = data_qian['FILTER']
    x = x[band=='g']
    y = y[band=='g']
    yerr = yerr[band=='g']
    ra = ra_qian[dbid==dbid_qian]
    dec = dec_qian[dbid==dbid_qian]
    # Catalog
    c = SkyCoord(ra, dec, unit=u.deg)
    idx, d2d, d3d = c.match_to_catalog_sky(catalog_yue)
    sep_constraint = d2d < 1*u.arcsec
    mask = idx[sep_constraint]
    # Match to Yue's catalog
    if len(data_yue['LOGBH'][mask]) == 0:
        return None
    mass = data_yue['LOGBH'][mask][0]
    mass_err = data_yue['LOGBH_ERR'][mask][0]
    z = data_yue['REDSHIFT'][mask][0]
    if mass == 0:
        return None
    log_L5100 = data_yue['LOGL5100'][mask][0]
    log_L5100err = data_yue['LOGL5100_ERR'][mask][0]
    if log_L5100 == 0:
        log_L5100 = np.log10((10**data_yue['LOGLBOL'][mask][0])/9.26)
        log_L5100err = data_yue['LOGLBOL_ERR'][mask][0]
    rm = False
    Lref = 'Shen et al. (2011)'
    LCref = 'Yang et al. (2020)'
    massref = 'Shen et al. (2011)'
    band = 'g'
    unit = 'mag'
    return dbid, ra[0], dec[0], mass, mass_err, z, log_L5100, log_L5100err, band, unit, x, y, yerr, rm, LCref, Lref, massref

# Du et al high Eddington rate BHs https://iopscience.iop.org/article/10.3847/1538-4357/aaae6b#apjaaae6bt3
def gather_data_du():
    # NONE OF THESE ARE ACCEPTABLE
    t = Table.read("data/apjaaae6bt3_mrt.txt", format="ascii.cds")
    name = t['Name']
    x_du = t['JDphot'].data
    y_du = t['rmag'].data
    yerr_du = t['e_rmag'].data
    out = []
    for namei in np.unique(name):
        x = x_du[name==namei]
        y = y_du[name==namei]
        yerr = yerr_du[name==namei]
        # From Table 6
        if namei.startswith("SDSS J074352"):
            mass = 7.93
            masserr = np.log10(np.sqrt((10**0.05)**2 + (10**0.04)**2))
            z = 0.2520
            log_Mdot = 1.69
            log_mdoterr = np.log10(np.sqrt((10**0.12)**2 + (10**0.13)**2))
            log_L5100 = 45.37
            log_L5100err = 0.02
        elif namei.startswith("SDSS J075051"):
            mass = 7.67
            masserr = np.log10(np.sqrt((10**0.11)**2 + (10**0.07)**2))
            z = 0.4004
            log_Mdot = 2.14
            log_mdoterr = np.log10(np.sqrt((10**0.16)**2 + (10**0.24)**2))
            log_L5100 = 45.33
            log_L5100err = 0.01
        elif namei.startswith("SDSS J075101"):
            mass = 7.20
            masserr = np.log10(np.sqrt((10**0.08)**2 + (10**0.12)**2))
            z = 0.1209
            log_Mdot = 1.30
            log_mdoterr = np.log10(np.sqrt((10**0.24)**2 + (10**0.24)**2))
            log_L5100 = 44.12
            log_L5100err = 0.05
        elif namei.startswith("SDSS J075949"):
            mass = 7.21
            masserr = np.log10(np.sqrt((10**0.16)**2 + (10**0.19)**2))
            z = 0.1879
            log_Mdot = 0.90
            log_mdoterr = np.log10(np.sqrt((10**0.56)**2 + (10**0.56)**2))
            log_L5100 = 44.20
            log_L5100err = 0.03
        elif namei.startswith("SDSS J081441"):
            mass = 7.22
            masserr = np.log10(np.sqrt((10**0.10)**2 + (10**0.11)**2))
            z = 0.1626
            log_Mdot = 1.14
            log_mdoterr = np.log10(np.sqrt((10**0.51)**2 + (10**0.52)**2))
            log_L5100 = 44.01
            log_L5100err = 0.07
        elif namei.startswith("SDSS J083553"):
            mass = 6.87
            masserr = np.log10(np.sqrt((10**0.16)**2 + (10**0.25)**2))
            z = 0.2051
            log_Mdot = 2.41
            log_mdoterr = np.log10(np.sqrt((10**0.53)**2 + (10**0.35)**2))
            log_L5100 = 44.44
            log_L5100err = 0.02
        elif namei.startswith("SDSS J084533"):
            mass = 6.82
            masserr = np.log10(np.sqrt((10**0.14)**2 + (10**0.10)**2))
            z = 0.3024
            log_Mdot = 2.77
            log_mdoterr = np.log10(np.sqrt((10**0.35)**2 + (10**0.34)**2))
            log_L5100 = 44.54
            log_L5100err = 0.04
        elif namei.startswith("SDSS J093302"):
            mass = 7.08
            masserr = np.log10(np.sqrt((10**0.08)**2 + (10**0.11)**2))
            z = 0.1772
            log_Mdot = 1.79
            log_mdoterr = np.log10(np.sqrt((10**0.40)**2 + (10**0.40)**2))
            log_L5100 = 44.31
            log_L5100err = 0.13
        elif namei.startswith("SDSS J100402"):
            mass = 7.44
            masserr = np.log10(np.sqrt((10**0.37)**2 + (10**0.06)**2))
            z = 0.3272
            log_Mdot = 2.89
            log_mdoterr = np.log10(np.sqrt((10**0.13)**2 + (10**0.75)**2))
            log_L5100 = 45.52
            log_L5100err = 0.01
        elif namei.startswith("SDSS J101000"):
            mass = 7.46
            masserr = np.log10(np.sqrt((10**0.27)**2 + (10**0.14)**2))
            z = 0.2564
            log_Mdot = 1.70
            log_mdoterr = np.log10(np.sqrt((10**0.31)**2 + (10**0.56)**2))
            log_L5100 = 44.76
            log_L5100err = 0.02
        massref = "Du et al. (2018)"
        Lref = "Du et al. (2018)"
        LCref = "Du et al. (2018)"
        band = 'r'
        unit = 'mag'
        c = SkyCoord.from_name(namei)
        out.append([namei, c.ra.deg, c.dec.deg, mass, abs(masserr), z, log_L5100, log_L5100err, band, unit, np.array(x[y>0]), np.array(y[y>0]), np.array(yerr[y>0]), True, LCref, Lref, massref])
    return out

# Bentz 2014 THE MASS OF THE CENTRAL BLACK HOLE IN THE NEARBY SEYFERT GALAXY NGC 5273
def gather_data_5273():
    # NOT ACCEPTABLE
    x, y, yerr = np.loadtxt("data/apj502545t1_ascii.txt", unpack=True)
    z = 0.00362
    LCref = "Bentz et al. (2014)"
    massref = "Bentz et al. (2014)"
    mass = np.log10(4.7e6)
    masserr = 1.6e6
    masserr = 0.434*masserr/(10**mass)
    namei = "NGC 5273"
    band = 'g'
    unit = 'mag'
    log_Mdot = -2.50 # log Mdot/MSun
    log_mdoterr = np.log10(np.sqrt((10**1.33)**2 + (10**0.67)**2))
    log_L5100 = 41.54 # erg/s
    log_L5100err = 0.15
    Lref = "Bentz et al. (2014)"
    c = SkyCoord.from_name(namei)
    return  np.array([namei, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, x, y, yerr, True, LCref, Lref, massref])

# A REVERBERATION-BASED BLACK HOLE MASS FOR MCG-06-30-15
def gather_data_bentz16b():
    # NOT ACCEPTABLE
    x, y, yerr = np.loadtxt("data/bentz2016b.txt", unpack=True)
    z = 0.0077
    LCref = "Bentz et al. (2016)"
    massref = "Bentz et al. (2016)"
    band = 'V'
    unit = r'$10^{-15}$ erg s$^{-1}$ cm$^{-2} \AA^{-1}$'
    mass = np.log10(1.6e6)
    masserr = 0.4e6
    masserr = 0.434*masserr/(10**mass)
    namei = "MCG-06-30-15"
    log_Mdot = -1.28 # log Mdot/MSun
    log_mdoterr = np.log10(np.sqrt((10**0.58)**2 + (10**0.73)**2))
    log_L5100 = 41.65 # erg/s
    log_L5100err = 0.25
    Lref = "Bentz et al. (2016)"
    c = SkyCoord.from_name(namei)
    return  np.array([namei, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, x, y, yerr, True, LCref, Lref, massref])

# A REVERBERATION-BASED BLACK HOLE MASS FOR UGC 06728
def gather_data_bentz16a():
    # NOT ACCEPTABLE
    x, y, yerr = np.loadtxt("data/bentz2016a.txt", unpack=True)
    z = 0.00652
    LCref = "Bentz et al. (2016)"
    massref = "Bentz et al. (2016)"
    band = 'g'
    unit = 'mag'
    mass = np.log10(7.1e5)
    masserr = 4.0e5
    masserr = 0.434*masserr/(10**mass)
    namei = "UGC 06728"
    log_L5100 = 41.86 # erg/s
    log_L5100err = 0.08
    Lref = "Bentz et al. (2016)"
    c = SkyCoord.from_name(namei)
    return np.array([namei, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, x, y, yerr, True, LCref, Lref, massref])

# Kepler LC https://iopscience.iop.org/article/10.1088/0004-637X/795/1/38/pdf
def gather_data_kepler():
    # NOT ACCEPTABLE
    t = Table.read("data/apj501630t2_mrt.txt", format="ascii.cds")
    name = "KA1858+4850"
    x = t['HJD'].data
    y = t['Vmag'].data
    yerr = t['e_Vmag'].data
    z = 0.078
    LCref = "Pie et al. (2014)"
    massref = "Pie et al. (2014)"
    mass = np.log10(8.06e6)
    masserr1 = 1.59e6
    masserr2 = 1.72e6
    masserr = np.sqrt(masserr1**2 + masserr2**2)
    masserr = 0.434*masserr/(10**mass)
    log_L5100 = np.log10(1.64e43) # erg/s
    log_L5100err = np.sqrt(0.59e43**2 + 0.65e43**2)
    log_L5100err = 0.434*log_L5100err/(10**log_L5100)
    Lref = "Pie et al. (2014)"
    band = 'V'
    unit = 'mag'
    c = SkyCoord.from_name("1RXSJ185800.9+485020")
    return np.array([name, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, x, y, yerr, True, LCref, Lref, massref])

# Kepler LC https://arxiv.org/abs/1103.2153
def gather_data_kepler2():
    # NOT ACCEPTABLE
    t = Table.read("data/apj387310t2_mrt.txt", format="ascii.cds")
    name = "Zw 229-015"
    x = t['HJD'].data
    y = t['Vmag'].data
    yerr = t['e_Vmag'].data
    # 
    t = Table.read("data/apjaaf806t4_mrt.txt", format="ascii.cds")
    mask_obj = t['AGN']=="Zw229-015"
    x = np.append(x, t['HJD'].data[mask_obj])
    y = np.append(y, t['Vmag'].data[mask_obj])
    yerr = np.append(yerr, t['e_Vmag'].data[mask_obj])
    z = 0.0273
    LCref = "Barth et al. (2011)"
    massref = "Barth et al. (2011)"
    band = 'V'
    unit = 'mag'
    mass = np.log10(1.00e7)
    masserr1 = 0.19e7
    masserr2 = 0.24e7
    masserr = np.sqrt(masserr1**2 + masserr2**2)
    masserr = 0.434*masserr/(10**mass)
    log_L5100 = np.log10(7.1e42) # erg/s
    log_L5100err = 0
    Lref = "Barth et al. (2011)"
    c = SkyCoord.from_name("KIC 006932990")
    return np.array([name, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, x, y, yerr, True, LCref, Lref, massref])

# SDSS-RM
# https://iopscience.iop.org/article/10.3847/1538-4357/aa98dc
def gather_data_grier17():
    # LIGHT CURVES ARE TERRIBLE
    out = []
    t1 = Table.read("data/apjaa98dct1_mrt.txt", format="ascii.cds")
    rmid = t1['RMID']
    ra = t1['RAdeg']
    dec =  t1['DEdeg']
    name = t1['SDSS']
    z = t1['z']
    # light curve data
    t2 = Table.read("data/apjaa98dct2_mrt.txt", format="ascii.cds")
    t2 = t2[t2['Band']=='g']
    rmid_lc = t2['RMID']
    x = t2['MJD'].data
    y = t2['Flux'].data
    yerr = t2['e_Flux'].data
    # Fixed BH masses are in the errata: https://iopscience.iop.org/article/10.3847/1538-4357/aaee67
    # error bars are wrong, so use a white noise term
    rmid_Hb, mass_Hb, masserr1_Hb, masserr2_Hb = np.loadtxt("data/apjaaee67t1_ascii.txt", unpack=True, dtype=str)
    rmid_Ha, mass_Ha, masserr1_Ha, masserr2_Ha = np.loadtxt("data/apjaaee67t1_ascii.txt", unpack=True, dtype=str)
    mass_Hb = mass_Hb.astype(np.float)
    mass_Ha = mass_Ha.astype(np.float)
    masserr1_Hb = masserr1_Hb.astype(np.float)
    masserr2_Hb = masserr2_Hb.astype(np.float)
    masserr1_Ha = masserr1_Ha.astype(np.float)
    masserr2_Ha = masserr2_Ha.astype(np.float)
    rmid_Hb = [int(i[2:]) for i in rmid_Hb]
    rmid_Ha = [int(i[2:]) for i in rmid_Ha]
    for i, rmid_i in enumerate(rmid):
        idx_Hb = (rmid_i==rmid_Hb)
        idx_Ha = (rmid_i==rmid_Ha)
        idx_lc = (rmid_i==rmid_lc)
        if np.any(idx_Hb):
            mass = mass_Hb[idx_Hb]*1e7
            masserr = np.sqrt((masserr1_Hb*1e7)**2 + (masserr2_Hb*1e7)**2)
            masserr = 0.434*masserr[idx_Hb]/(mass)
        elif np.any(idx_Ha):
            mass = mass_Ha[idx_Ha]*1e7
            masserr = np.sqrt((masserr1_Ha*1e7)**2 + (masserr2_Ha*1e7)**2)
            masserr = 0.434*masserr[idx_Ha]/(mass)
        else:
            continue
        ref = "Grier2018"
        mass = np.log10(mass)
        log_L5100 = 0
        log_L5100err = 0
        log_Mdot = 0
        log_mdoterr = 0
        Lref = ""
        #out.append([name[i], ra[i], dec[i], mass[0], masserr[0], z[i], log_L5100, log_L5100err, ref, x[idx_lc], y[idx_lc], yerr[idx_lc], True, Lref])
    return out

# https://iopscience.iop.org/article/10.1088/0004-637X/795/2/149#apj502376t1
def gather_data_peterson14():
    # NOT ACCEPTABLE
    t = Table.read("data/apj502376t1_mrt.txt", format="ascii.cds")
    x = t['HJD'].data
    y = t['FC'].data
    yerr = t['e_FC'].data
    z = 0.01632
    LCref = "Peterson et al. (2014)"
    mass = np.log10(9.57e6)
    masserr = 1.03e6
    masserr = 0.434*masserr/(10**mass)
    namei = "NGC 7469"
    log_Mdot = -0.46 # log Mdot/MSun
    log_mdoterr = np.log10(np.sqrt((10**0.26)**2 + (10**0.42)**2))
    log_L5100 = 0 # erg/s
    log_L5100err = 0
    Lref = "Peterson et al. (2014)"
    massref = "Peterson et al. (2014)"
    band = 'V'
    unit = r'$10^{-15}$ erg s$^{-1}$ cm$^{-2} \AA^{-1}$'
    c = SkyCoord.from_name(namei)
    return np.array([namei, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, x, y, yerr, True, LCref, Lref, massref])

# https://arxiv.org/abs/1607.08060 NGC 5548 RM
def gather_data_lu16():
    # NOT ACCEPTABLE
    t = Table.read("data/apjaa2763t1_mrt.txt", format="ascii.cds")
    namei = "NGC 5548"
    x = t['MJD1'].data
    y = t['F5100'].data
    yerr = t['e_F5100'].data
    z = 0.01627
    LCref = "Lu et al. (2016)"
    massref = "Lu et al. (2016)"
    band = '5100 AA'
    mass = np.log10(8.71e7)
    masserr1 = 3.21e7
    masserr2 = 2.61e7
    masserr = np.sqrt(masserr1**2 + masserr2**2)
    masserr = 0.434*masserr/(10**mass)
    log_Mdot = -1.60 # log Mdot/MSun
    log_mdoterr = np.log10(np.sqrt((10**0.46)**2 + (10**0.49)**2))
    log_L5100 = 43.21 # erg/s
    log_L5100err = 0.12
    Lref = ""
    unit = r'$10^{-15}$ erg s$^{-1}$ cm$^{-2} \AA^{-1}$'
    c = SkyCoord.from_name(namei)
    return np.array([namei, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, x, y, yerr, True, LCref, Lref, massref])

# https://ui.adsabs.harvard.edu/abs/2020ApJ...890...71H/abstract
def gather_data_hu20():
    t = Table.read("data/apjab6a17t1_mrt.txt", format="ascii.cds")
    x = t['JD'].data
    y = t['F5100'].data*1e15
    yerr = t['e_F5100'].data*1e15
    y_med = np.median(y)
    y = y/y_med
    yerr = yerr/y_med
    z = 0.063
    mass = np.log10(0.97e7)
    masserr1 = 0.15e7
    masserr2 = 0.18e7
    masserr = np.sqrt(masserr1**2 + masserr2**2)
    masserr = 0.434*masserr/(10**mass)
    namei = "PG 2130+099"
    log_Mdot = 1.40 # log Mdot/MSun
    log_mdoterr = np.log10(np.sqrt((10**0.24)**2 + (10**0.19)**2))
    log_L5100 = np.log10(2.5e44) # erg/s
    log_L5100err = 0.434*np.log10(0.12e44)/2.5e44
    LCref = "Hu et al. (2020)"
    Lref = "Hu et al. (2020)"
    massref = "Hu et al. (2020)"
    c = SkyCoord.from_name(namei)
    band = 'V'
    unit = r'$10^{-15}$ erg s$^{-1}$ cm$^{-2} \AA^{-1}$'
    return np.array([namei, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, x, y, yerr, True, LCref, Lref, massref])


# https://arxiv.org/abs/1610.00008
# Fausnaugh light curves
def gather_data_fausnaugh():
    out = []
    for filename in glob.glob("data/apjaa6d52t*_mrt.txt"):
        t = Table.read(filename, format="ascii.cds")
        x = t['HJD'].data
        y = t['Flambda'].data
        yerr = t['e_Flambda'].data
        table = int(filename.split("_mrt")[0][-1])
        band = 'V'
        unit = r'$10^{-15}$ erg s$^{-1}$ cm$^{-2} \AA^{-1}$'
        # Use Hb 
        if table==5:
            namei = "MCG+08-11-011"
            z = 0.0205
            mass = 7.45
            masserr = 0.47
            log_Mdot = -0.96 # log Mdot/MSun
            log_mdoterr = np.log10(np.sqrt((10**0.25)**2 + (10**0.28)**2))
            log_L5100 = 43.33 # erg/s
            log_L5100err = 0.11
        elif table==6:
            namei = "NGC 2617"
            z = 0.0142
            mass = 7.51
            masserr = 0.47
            log_Mdot = -1.98
            log_mdoterr = np.log10(np.sqrt((10**0.55)**2 + (10**0.51)**2))
            log_L5100 = 42.67 
            log_L5100err = 0.16
        elif table==7:
            namei = "NGC 4051"
            z = 0.0023
            mass = 5.67
            masserr = 0.47
            log_Mdot = 0.99
            log_mdoterr = np.log10(np.sqrt((10**1.11)**2 + (10**-1.06)**2))
            log_L5100 = 41.96
            log_L5100err = 0.19
        elif table==8:
            namei = "3C 382"
            z = 0.0579
            mass = 8.98
            masserr = 0.47
            log_Mdot = -2.09
            log_mdoterr = np.log10(np.sqrt((10**0.26)**2 + (10**0.35)**2))
            log_L5100 = 43.84
            log_L5100err = 0.10
            band = 'g'
        elif table==9:
            namei = "Mrk 374"
            z = 0.0426
            mass = 7.32
            masserr = 0.54
            log_Mdot = -0.56
            log_mdoterr = np.log10(np.sqrt((10**0.30)**2 + (10**0.36)**2))
            log_L5100 = 43.77
            log_L5100err = 0.04
            band = 'g'
        LCref = "Fausnaugh et al. (2017)"
        Lref = "Fausnaugh et al. (2017)"
        massref = "Fausnaugh et al. (2017)"
        c = SkyCoord.from_name(namei)
        out.append([namei, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, x, y, yerr, True, LCref, Lref, massref])
    return out

from astropy.table import Table
# https://www.physics.uci.edu/~barth/lamp.html
# Walsh 2009 https://iopscience.iop.org/article/10.1088/0067-0049/185/1/156
def gather_data_barth():
    # NONE OF THESE ARE ACCEPTABLE
    t = Table.read("data/apjs322809t6_mrt.txt", format="ascii.cds")
    name = t['Name']
    x_barth = t['HJD-V'].data.data
    y_barth = t['Vmag'].data.data
    yerr_barth = t['e_Vmag'].data.data
    band = 'V'
    unit = 'mag'
    # Masses are from Bentz 2009 https://iopscience.iop.org/article/10.1088/0004-637X/705/1/199#apj322808t13
    out = []
    for namei in np.unique(t['Name']):
        x = x_barth[name==namei]
        y = y_barth[name==namei]
        yerr = yerr_barth[name==namei]
        if namei=="Arp 151":
            mass = np.log10(6.72e6)
            masserr = np.sqrt((0.96*1e6)**2 + (1.24*1e6)**2)
            masserr = 0.434*masserr/(10**mass)
            z = 0.02109
            log_Mdot = -0.44
            log_mdoterr = np.log10(np.sqrt((10**0.30)**2 + (10**0.28)**2))
            log_L5100 = 42.55
            log_L5100err = 0.10
        elif namei=="Mrk 142":
            mass = np.log10(2.17e6)
            masserr = np.sqrt((0.77*1e6)**2 + (0.83*1e6)**2)
            masserr = 0.434*masserr/(10**mass)
            z = 0.04494
            log_Mdot = 1.90
            log_mdoterr = np.log10(np.sqrt((10**0.85)**2 + (10**0.86)**2))
            log_L5100 = 43.56
            log_L5100err = 0.06
        elif namei=="SBS 1116": # SBS 1116+583A
            namei = "J111857.7+580324"
            mass = np.log10(5.8e6)
            masserr = np.sqrt((2.09*1e6)**2 + (1.86*1e6)**2)
            masserr = 0.434*masserr/(10**mass)
            z = 0.02787
            log_Mdot = -0.87
            log_mdoterr = np.log10(np.sqrt((10**0.51)**2 + (10**0.71)**2))
            log_L5100 = 42.14
            log_L5100err = 0.23
        elif namei=="Mrk 1310":
            mass = np.log10(2.24e6)
            masserr = np.sqrt((0.9*1e6)**2 + (0.9*1e6)**2)
            masserr = 0.434*masserr/(10**mass)
            z = 0.01941
            log_Mdot = -0.31
            log_mdoterr = np.log10(np.sqrt((10**0.35)**2 + (10**0.39)**2))
            log_L5100 = 42.29
            log_L5100err = 0.14
        elif namei=="Mrk 202":
            mass = np.log10(1.42e6)
            masserr = np.sqrt((0.85*1e6)**2 + (0.59*1e6)**2)
            masserr = 0.434*masserr/(10**mass)
            z = 0.02102
            log_Mdot = 0.66
            log_mdoterr = np.log10(np.sqrt((10**0.59)**2 + (10**0.65)**2))
            log_L5100 = 42.26
            log_L5100err = 0.14
        elif namei=="NGC 4253":
            mass = np.log10(1.76e6)
            masserr = np.sqrt((1.56*1e6)**2 + (1.40*1e6)**2)
            masserr = 0.434*masserr/(10**mass)
            z = 0.01293
            log_Mdot = 0.36
            log_mdoterr = np.log10(np.sqrt((10**0.36)**2 + (10**0.42)**2))
            log_L5100 = 42.57
            log_L5100err = 0.12
        elif namei=="NGC 4748":
            mass = np.log10(2.57e6)
            masserr = np.sqrt((1.30*1e6)**2 + (1.25*1e6)**2)
            masserr = 0.434*masserr/(10**mass)
            z = 0.01463
            log_Mdot = 0.10
            log_mdoterr = np.log10(np.sqrt((10**0.61)**2 + (10**0.44)**2))
            log_L5100 = 42.56
            log_L5100err = 0.12
        elif namei=="NGC 5548":
            mass = np.log10(82e6)
            masserr = np.sqrt((20*1e6)**2 + (28*1e6)**2)
            masserr = 0.434*masserr/(10**mass)
            z = 0.01718
            log_Mdot = -0.16
            log_mdoterr = np.log10(np.sqrt((10**0.46)**2 + (10**0.49)**2))
            log_L5100 = 43.39
            log_L5100err = 0.10
        elif namei=="NGC 6814":
            mass = np.log10(18.5e6)
            masserr = np.sqrt((3.5*1e6)**2 + (3.5*1e6)**2)
            masserr = 0.434*masserr/(10**mass)
            z = 0.00521
            log_Mdot = -1.64
            log_mdoterr = np.log10(np.sqrt((10**0.46)**2 + (10**0.80)**2))
            log_L5100 = 42.12
            log_L5100err = 0.28
        elif namei=="Mrk 290":
            mass = 7.277
            masserr = 0.061
            z = 0.02958
            log_Mdot = -0.85
            log_mdoterr = np.log10(np.sqrt((10**0.23)**2 + (10**0.23)**2))
            log_L5100 = 43.17
            log_L5100err = 0.06
        c = SkyCoord.from_name(namei)
        LCref = "Walsh et al. (2009)"
        Lref = "Walsh et al. (2009)"
        massref = "Bentz et al. (2009)"
        # Mdot = mdot c^2 / LEdd
        out.append([namei, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, x, y, yerr, True, LCref, Lref, massref])
    return out

# https://iopscience.iop.org/article/10.1086/308704
def gather_data_kapsi99(filepath):
    
    # Read Kapsi 99 tables
    name_kapsi, h_kapsi, m_kapsi, s_kapsi, d_kapsi, am_kapsi, as_kapsi, z_kapsi, mB_kapsi, \
        MB_kapsi, AB_kapsi, Nph_kapsi, Nsp_kapsi, f5100_kapsi, f5100err_kapsi, Rcomp_kapsi, PAcomp_kapsi \
        = np.genfromtxt('data/kapsi1.txt', unpack=True, dtype=str)
    coord_kapsi = concatenate([SkyCoord.from_name(i) for i in name_kapsi])
    
    x, y, yerr = np.loadtxt(filepath, unpack=True)
    for idx, name in enumerate(name_kapsi):
        log_L5100 = 0
        log_L5100err = 0
        band = 'B' # I think some are R but not listed
        unit = r'$10^{-16}$ erg s$^{-1}$ cm$^{-2} \AA^{-1}$'
        if name.split('+')[0] in filepath:
            if "PG0026" in name:
                mass = 5.4e7
                masserr = np.sqrt((1.0e7)**2 + (1.1e7)**2)
            elif "PG0052" in name:
                mass = 22.0e7
                masserr = np.sqrt((6.3e7)**2 + (5.3e7)**2)
            elif "PG0804" in name:
                mass = 18.9e7
                masserr = np.sqrt((1.9e7)**2 + (1.7e7)**2)
            elif "PG0844" in name:
                mass = 2.16e7
                masserr = np.sqrt((0.9e7)**2 + (0.83e7)**2)
            elif "PG0953" in name:
                mass = 18.4e7
                masserr = np.sqrt((2.8e7)**2 + (3.4e7)**2)
            elif "PG1211" in name:
                mass = 4.05e7
                masserr = np.sqrt((0.96e7)**2 + (1.21e7)**2)
            elif "PG1226" in name:
                mass = 55.0e7
                masserr = np.sqrt((8.9e7)**2 + (7.9e7)**2)
            elif "PG1229" in name:
                mass = 7.5e7
                masserr = np.sqrt((3.6e7)**2 + (3.5e7)**2)
            elif "PG1307" in name:
                mass = 28e7
                masserr = np.sqrt((11e7)**2 + (18e7)**2)
            elif "PG1351" in name:
                mass = 4.6e7
                masserr = np.sqrt((3.2e7)**2 + (1.9e7)**2)
                log_L5100 = np.log10(5.50e44)
                log_L5100err = 0.434*0.49e44/5.50e44
            elif "PG1411" in name:
                mass = 8.0e7
                masserr = np.sqrt((3.0e7)**2 + (2.9e7)**2)
            elif "PG1426" in name:
                mass = 47e7
                masserr = np.sqrt((16e7)**2 + (20e7)**2)
            elif "PG1613" in name:
                mass = 24.1e7
                masserr = np.sqrt((12.5e7)**2 + (8.9e7)**2)
            elif "PG1617" in name:
                mass = 27.3e7
                masserr = np.sqrt((8.3e7)**2 + (9.7e7)**2)
            elif "PG1700" in name:
                mass = 6e7
                masserr = np.sqrt((13e7)**2 + (13e7)**2)
            elif "PG1704" in name:
                mass = 3.7e7
                masserr = np.sqrt((3.1e7)**2 + (4.0e7)**2)
            elif "PG2130" in name:
                mass = 14.4e7
                masserr = np.sqrt((5.1e7)**2 + (1.7e7)**2)
                log_L5100 = np.log10(2.16e44)
                log_L5100err = 0.434*0.20e44/2.16e44

            masserr = 0.434*masserr/(mass)
            mass = np.log10(mass)
            z = float(z_kapsi[idx])
            c = coord_kapsi[idx]
            LCref = "Kapsi et al. (1999)"
            Lref = "Kapsi et al. (1999)"
            massref = "Kapsi et al. (1999)"
            name = name[:2] + ' '+ name[2:]
            return np.array([name, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, x, y, yerr, True, LCref, Lref, massref])
        
# Mrk 142 https://iopscience.iop.org/article/10.3847/1538-4357/ab91b5#apjab91b5f1
def gather_data_crackett20():
    t = Table.read("data/apjab91b5t2_mrt.txt", format="ascii.cds")
    name = "Mrk 142"
    band = 'g'
    t = t[t['Filter']=='g']
    x = t['MJD']
    y = t['Flux']
    yerr = t['e_Flux']
    unit = 'arbitrary unit' # normalized to mean flux of 1
    c = SkyCoord.from_name(name)
    # Use mass from LAMP papers
    mass = 6.33646
    masserr = 0.226433
    z = 0.045
    DL = ((201.5*u.Mpc).to(u.cm)).value
    F5100 = 8.3e-16
    log_L5100 = np.log10(F5100*5100*4*np.pi*DL**2)
    log_L5100err = 0
    LCref = "Crackett et al. (2020)"
    Lref = "Crackett et al. (2020)"
    massref = "Crackett et al. (2020)"
    return np.array([name, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, x, y, yerr, True, LCref, Lref, massref])

# https://iopscience.iop.org/article/10.1088/0004-637X/721/1/715#apj365393t7
def gather_data_denney10():
    # NOT ACCEPTABLE
    name, z, _, _, _, _, _, _, _, _, _, _, _, _, _, _, mass, masserr1, masserr2, logL5100, logL5100err1, logL5100err2 = np.loadtxt("data/denneytarg.txt", unpack=True, dtype=str)
    name = np.array([i.lower().strip() for i in name])
    z = z.astype(np.float)
    mass = mass.astype(np.float)*1e6
    masserr1 = masserr1.astype(np.float)
    masserr2 = masserr2.astype(np.float)
    masserr =  np.sqrt((masserr1*1e6)**2 + (masserr2*1e6)**2)
    masserr = 0.434*masserr/(mass)
    mass = np.log10(mass)
    logL5100 = np.array(logL5100.astype(np.float))
    logL5100err1 = logL5100err1.astype(np.float)
    logL5100err2 = logL5100err2.astype(np.float)
    logL5100err = np.sqrt((10**logL5100err1)**2 + (10**logL5100err2)**2)
    logL5100err = np.array(0.434*logL5100err/(10**logL5100))
    out = []
    band = 'V'
    unit = r'$10^{-15}$ erg s$^{-1}$ cm$^{-2} \AA^{-1}$'
    for filename in glob.glob("data/denney_*.txt"):
        x, y, yerr = np.loadtxt(filename, unpack=True)
        namei = filename.split('_')[1].split('.txt')[0]
        mask_name = (name==namei)
        c = SkyCoord.from_name(namei)
        if 'ngc' in namei:
            namei = 'NGC '+namei.split('ngc')[1]
        elif 'mrk' in namei:
            namei = 'Mrk '+namei.split('mrk')[1]
        LCref = 'Denney et al. (2010)'
        massref = 'Denney et al. (2010)'
        Lref = 'Denney et al. (2010)'
        out.append([namei, c.ra.deg, c.dec.deg, mass[mask_name][0], masserr[mask_name][0], z[mask_name][0], logL5100[mask_name][0], logL5100err[mask_name][0], band, unit, x, y, yerr, True, LCref, Lref, massref])
    return np.array(out)

# https://iopscience.iop.org/article/10.3847/1538-4357/aadd11#apjaadd11t6
def gather_data_derosa():
    out = []
    LCref = "De Rosa et al. (2018)"
    Lref = "De Rosa et al. (2018)"
    massref = "De Rosa et al. (2018)"
    band = 'V'
    unit = r'$10^{-15}$ erg s$^{-1}$ cm$^{-2} \AA^{-1}$'
    t = Table.read("data/apjaadd11t5_mrt.txt", format="ascii.cds")
    x = np.array(t['D-Mrk704'].data.data)
    y = np.array(t['F-Mrk704'].data)
    yerr = np.array(t['e_F-Mrk704'].data)
    mass = 7.63
    masserr = 0.14
    z = 0.0292
    logL5100 = 43.72
    logL5100err = 0
    name = "Mrk 704"
    c = SkyCoord.from_name(name)
    out.append([name, c.ra.deg, c.dec.deg, mass, masserr, z, logL5100, logL5100err, band, unit, x[y<999], y[y<999], yerr[y<999], True, LCref, Lref, massref])
    #
    x = np.array(t['D-NGC3227-1'].data.data)
    y = np.array(t['F-NGC3227-1'].data)
    yerr = np.array(t['e_F-NGC3227-1'].data)
    mass = 6.57
    masserr = 0.13
    z = 0.0038
    logL5100 = 42.48
    logL5100err = 0
    name = "NGC 3227"
    c = SkyCoord.from_name(name)
    out.append([name, c.ra.deg, c.dec.deg, mass, masserr, z, logL5100, logL5100err, band, unit, x[y<999], y[y<999], yerr[y<999], True, LCref, Lref, massref])
    #
    x = np.array(t['D-NGC3227-2'].data.data)
    y = np.array(t['F-NGC3227-2'].data)
    yerr = np.array(t['e_F-NGC3227-2'].data)
    mass = 6.66
    masserr = 0.24
    z = 0.0038
    logL5100 = 42.48
    logL5100err = 0
    name = "NGC 3227"
    c = SkyCoord.from_name(name)
    out.append([name, c.ra.deg, c.dec.deg, mass, masserr, z, logL5100, logL5100err, band, unit, x[y<999], y[y<999], yerr[y<999], True, LCref, Lref, massref])
    #
    x = np.array(t['D-NGC3516'].data.data)
    y = np.array(t['F-NGC3516'].data)
    yerr = np.array(t['e_F-NGC3516'].data)
    mass = 7.63
    masserr = 0.13
    z = 0.0088
    logL5100 = 43.21
    logL5100err = 0
    name = "NGC 3516"
    c = SkyCoord.from_name(name)
    out.append([name, c.ra.deg, c.dec.deg, mass, masserr, z, logL5100, logL5100err, band, unit, x[y<999], y[y<999], yerr[y<999], True, LCref, Lref, massref])
    #
    x = np.array(t['D-NGC4151'].data.data)
    y = np.array(t['F-NGC4151'].data)
    yerr = np.array(t['e_F-NGC4151'].data)
    mass = 7.33
    masserr = 0.13
    z = 0.0033
    logL5100 = 42.37
    logL5100err = 0
    name = "NGC 4151"
    c = SkyCoord.from_name(name)
    out.append([name, c.ra.deg, c.dec.deg, mass, masserr, z, logL5100, logL5100err, band, unit, x[y<999], y[y<999], yerr[y<999], True, LCref, Lref, massref])
    #
    x = np.array(t['D-NGC5548'].data.data)
    y = np.array(t['F-NGC5548'].data)
    yerr = np.array(t['e_F-NGC5548'].data)
    mass = 7.39
    masserr = 0.14
    z = 0.0171
    logL5100 = 43.20
    logL5100err = 0
    name = "NGC 5548"
    c = SkyCoord.from_name(name)
    out.append([name, c.ra.deg, c.dec.deg, mass, masserr, z, logL5100, logL5100err, band, unit, x[y<999], y[y<999], yerr[y<999], True, LCref, Lref, massref])
    return out

# https://arxiv.org/pdf/2004.11295.pdf
def gather_data_cann():
    data = fits.getdata('data/local_rm/can2020agn.fits')
    x = data['mjd'][data['filtercode']=='zr']
    y = data['mag'][data['filtercode']=='zr']
    yerr = data['magerr'][data['filtercode']=='zr']
    name = 'SDSS J105621.45+313822.1'
    c = SkyCoord.from_name(name)
    masserr = 0.434*(2.2-1.3)*1e6/(2.2e6)
    mass = np.log10(2.2e6)
    z = 0.161
    log_L5100 = 0
    log_L5100err = 0
    massref = "Cann (2020)"
    Lref = ""
    LCref = 'ZTF'
    band = 'r'
    unit = 'mag'
    name = 'J1056+3138'
    return np.array([name, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, x, y, yerr, True, LCref, Lref, massref])

def gather_data_rgg():
    
    out = []
    zs = [0.0459,0.0466,0.0327,0.0144,0.0011,0.0299,0.0410,0.0384,0.0395,0.0317,0.0274,0.0421,0.0288,0.0266,
          0.0289,0.0325,0.0222,0.0501,0.0250,0.0272,0.0349,0.0262,0.0230,0.0378,0.0287]
    masses = [5.7,5.4,4.9,6.1,5.0,5.2,5.4,5.7,5.1,5.2,5.9,6.1,6.8,6.7,5.2,5.2,6.1,6.6,5.3,6.0,5.2,6.4,6.6,5.4,5.5]
    logLHa = [39.38,40.15,39.41,40.13,38.15,39.73,39.67,40.16,39.82,39.45,39.52,40.67,40.10,39.56,39.26,40.09,39.56,39.97,39.39,39.99,39.58,39.51,39.75,40.42,38.88]
    coords = ['J024656.39-003304.80', 'J090613.75+561015.5', 'J095418.15+471725.1', 'J122342.82+581446.4', ' J122548.86+333248.7', 'J144012.70+024743.5', 'J085125.81+393541.7', 'J152637.36+065941.6', 'J153425.58+040806.6', 'J160531.84+174826.1', 'J004042.10-110957.7', 'J084029.91+470710.4', 'J090019.66+171736.9', 'J091122.24+615245.5', 'J101440.21+192448.9', 'J105100.64+655940.7', 'J105447.88+025652.4', 'J111548.27+150017.7', 'J112315.75+240205.1', 'J114343.77+550019.4', 'J120325.66+330846.1', 'J130724.64+523715.5', 'J131503.77+223522.7', 'J131603.91+292254.0', 'J134332.09+253157.7']
    names = ['1','9','11','20','21','32','48','119','123','127','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O']
    
    for i in range(1,25):
        
        f = 'data/rgg/%d.fits' % i
        
        # Load data
        data = fits.open(f)[1].data
        mjd_r_ztf = data['mjd'][data['filtercode']=='zr']
        mjd_g_ztf = data['mjd'][data['filtercode']=='zg']

        r_ztf = data['mag'][data['filtercode']=='zr']
        g_ztf = data['mag'][data['filtercode']=='zg']

        r_err_ztf = data['magerr'][data['filtercode']=='zr']
        g_err_ztf = data['magerr'][data['filtercode']=='zg']

        # Outlier rejection
        masserr = 0.3 # ROUGH ESTIMATE
        
        z = zs[i-1]
        mass = masses[i-1]
        
        # CONVERT L5100 FROM RGG EQUATION 2:
        LHa = 10**logLHa[i-1]
        L5100 = 1.1934e7*LHa**(1000/1157)
        
        log_L5100 = np.log10(L5100)
        log_L5100err = 0
        
        c = SkyCoord.from_name(coords[i-1])
        
        name='RGG %s' % names[i-1]
        if name=='RGG 21': # Skip NGC 4395!!
            continue
        
        massref = 'RGG (2015)'
        LCref = 'ZTF'
        band = 'r'
        unit = 'mag'
        Lref = 'RGG (2015)'
        
        out.append(np.array([name, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, mjd_r_ztf, r_ztf, r_err_ztf, False, LCref, Lref, massref]))
        
    return out

df_localrm = pd.read_csv('data/local_rm/mbh.csv', skiprows=2)
objs = np.array([i.lower() for i in df_localrm['Object']])

def gather_data_ztf_rm():
    
    out = []
    
    for f in glob.glob('data/local_rm/*.fits'):
        
        print(f)
        
        fname = os.path.basename(f.split('.')[0]).lower()
        if fname not in objs:
            continue
            
        print(fname)
            
        mass = df_localrm['log M_BH'].values[fname==objs][0]
        masserr1 = df_localrm['+err(log M_BH)'].values[fname==objs][0]
        masserr2 = df_localrm['-err(log M_BH)'].values[fname==objs][0]
        masserr = np.mean([masserr1,masserr2])
        z = df_localrm['Redshift'].values[fname==objs][0]
        
        massref = 'Bentz database'
        LCref = 'ZTF'
        band = 'r'
        unit = 'mag'
        
        # Other BH masses 
        # high accretion rate - RM
        # https://iopscience.iop.org/article/10.1088/0004-637X/806/1/22/pdf
        #S5 0836+71 EXCLUDE, BLAZAR
        
        if fname=='mrk1044':
            mass = 6.45
            masserr = 0.125
            massref = 'Du et al. (2015)'
            c = SkyCoord.from_name('Mrk 1044')
            dfname = 'Mrk 1044'
        elif fname=='mrk382':
            mass = 6.50
            masserr = 0.24
            massref = 'Du et al. (2015)'
            c = SkyCoord.from_name('Mrk 382')
            dfname = 'Mrk 382'
        elif fname=='MCG+06-26-012': # Not in ZTF I guess
            mass = 6.92
            masserr = 0.13
            massref = 'Du et al. (2015)'
            c = SkyCoord.from_name('MCG+06-26-012')
            dfname = 'MCG+06-26-012'
        elif fname=='WAS61': # IRAS F12397+3333
            mass = 6.79
            masserr = 0.36
            massref = 'Du et al. (2015)'
            c = SkyCoord.from_name('WAS61')
        elif fname=='PG1247+267':
            mass = np.log10(8.3e8)
            masserr = 0.434*3.05e8/8.3e8
            massref = 'Du et al. (2015)'
            c = SkyCoord.from_name('PG1247+267')
            dfname = 'PG 1247+267'
        elif fname=='ngc5940':
            continue # can't find a good BH mass for this
        elif fname=='mrk486':
            mass = 7.24
            masserr = 0.09
            massref = 'Du et al. (2015)'
            c = SkyCoord.from_name('Mrk 486')
            dfname = 'Mrk 486'
        elif fname=='mrk493':
            mass = 6.14
            masserr = 0.075
            massref = 'Du et al. (2015)'
            c = SkyCoord.from_name('Mrk 493')
            dfname = 'Mrk 493'
        elif fname=='arp120b':
            continue # Not in ZTF I guess
        
        #print(mass, masserr, z)
        Lref = ""
                
        # Load data
        data = fits.open(f)[1].data
        mjd_r_ztf = data['mjd'][data['filtercode']=='zr']
        mjd_g_ztf = data['mjd'][data['filtercode']=='zg']

        r_ztf = data['mag'][data['filtercode']=='zr']
        g_ztf = data['mag'][data['filtercode']=='zg']

        r_err_ztf = data['magerr'][data['filtercode']=='zr']
        g_err_ztf = data['magerr'][data['filtercode']=='zg']

        # Outlier rejection
        if fname=='ngc4253':
            log_L5100 = 42.51
            log_L5100err = 0.13
            Lref = 'Bentz et al. (2013)'
            c = SkyCoord.from_name('NGC 4253')
            dfname = 'NGC 4253'
        elif fname=='pg0844+349':
            log_L5100 = 44.24 # from https://www.aanda.org/articles/aa/pdf/2018/06/aa32220-17.pdf
            log_L5100err = 0.04 # see Kapsi 2000 ref within
            Lref = "Kapsi 2000"
            c = SkyCoord.from_name('PG 0844+349')
            dfname = 'PG 0844+349'
        elif fname=='mrk50':
            log_L5100 = 42.731
            log_L5100err = 0.005
            Lref = "Le et al. (2020)" # https://arxiv.org/pdf/2008.02990.pdf
            c = SkyCoord.from_name('Mrk 50')
            dfname = 'Mrk 50'
        elif fname=='arp151':
            log_L5100 = 42.48
            log_L5100err = 0.11
            Lref = 'Bentz et al. (2013)'
            c = SkyCoord.from_name('Arp 151')
            dfname = 'Arp 151'
        elif fname=='pg2130+099':
            log_L5100 = 44.14
            log_L5100err = 0.03
            Lref = 'Bentz et al. (2013)'
            c = SkyCoord.from_name('PG 2130+099')
            dfname = 'PG 2130+099'
        elif fname=='ngc4051':
            log_L5100 = 41.96
            log_L5100err = 0.20
            Lref = 'Bentz et al. (2013)'
            c = SkyCoord.from_name('NGC 4051')
            dfname = 'NGC 4051'
        elif fname=='mrk817':
            log_L5100 = 43.73
            log_L5100err = 0.05
            Lref = 'Bentz et al. (2013)'
            c = SkyCoord.from_name('Mrk 817')
            dfname = 'Mrk 817'
        elif fname=='mrk335':
            log_L5100 = 43.68
            log_L5100err = 0.06
            Lref = 'Bentz et al. (2013)'
            c = SkyCoord.from_name('Mrk 335')
            dfname = 'Mrk 335'
        elif fname=='pg1307+085':
            log_L5100 = 44.79
            log_L5100err = 0.02
            Lref = 'Bentz et al. (2013)'
            c = SkyCoord.from_name('PG 1307+085')
            dfname = 'PG 1307+085'
        elif fname=='mrk1044':
            log_L5100 = np.log10(0.47e43)
            log_L5100err = 0.434*0.1e43/(0.47e43)
            Lref = 'Bentz et al. (2013)' # really in Wang 2014 but not listed
            c = SkyCoord.from_name('Mrk 1044')
            dfname = 'Mrk 1044'
        else:
            # Additional objects
            log_L5100 = 0 
            log_L5100err = 0
            Lref = ''
            # Get names for ones without L
            if 'mrk' in fname:
                dfname = 'Mrk '+fname.split('mrk')[1]
            elif 'pg' in fname:
                dfname = 'PG '+fname.split('pg')[1]
            elif '3c' in fname:
                dfname = '3C '+fname.split('3c')[1]
            elif 'was' in fname:
                dfname = 'WAS '+fname.split('was')[1]
            elif 'ngc' in fname:
                dfname = 'NGC '+fname.split('ngc')[1]
            elif 'sbs' in fname:
                dfname = 'SBS '+fname.split('sbs')[1]
            else:
                dfname = fname.upper()
            c = SkyCoord.from_name(dfname)
            
        print(dfname)
                
        out.append(np.array([dfname, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, mjd_r_ztf, r_ztf, r_err_ztf, True, LCref, Lref, massref]))
        
    return out

# https://iopscience.iop.org/article/10.3847/1538-4357/aaf806#apjaaf806f2
def gather_data_pancoast():
    # THIS MIGHT BE A GOOD ONE TO ADD,
    # FIND BH masses
    from astropy.io import ascii
    data = ascii.read("data/apjaaf806t4_mrt.txt")
    
    out = []
    
    for name in np.unique(data['AGN']):
        mask = data['AGN']==name
        mjd = data[mask]['HJD'].data
        Vmag = data[mask]['Vmag'].data
        Vmagerr = data[mask]['e_Vmag'].data
        band = 'V'
        unit = 'mag'
        LCref = 'Pancoast et al. (2019)'
        massref = 'Bentz database'
                        
        # mass reference (Bentz DB) or Williams https://arxiv.org/pdf/1809.05113.pdf
        if name == 'Mrk 1511': # NGC 5940
            massref = 'Williams et al. (2018)'
            mass = 7.11
            masserr = np.mean([0.20, 0.17])
            log_L5100 = 42.16 # https://arxiv.org/pdf/1909.06735.pdf
            log_L5100err = 0.06 
            Lref = 'Du et al. (2019)'
            c = SkyCoord.from_name('Mrk 1511')
            z = 0.0339
        elif name == 'Mrk 279':
            mass = 7.435
            masserr = np.mean([0.099, 1.333])
            log_L5100 = 43.64
            log_L5100err = 0.08
            Lref = 'Bentz et al. (2013)'
            c = SkyCoord.from_name('Mrk 279')
            z = 0.0305
        elif name == 'Mrk 40': # Arp 151
            mass = 6.670
            masserr = np.mean([0.045, 0.054])
            log_L5100 = 42.48
            log_L5100err = 0.11
            Lref = 'Bentz et al. (2013)'
            c = SkyCoord.from_name('Mrk 40')
            z = 0.0211
        elif name == 'Mrk 50':
            mass = 7.422
            masserr = np.mean([0.057, 0.068])
            DL = ((102.5*u.Mpc).to(u.cm)).value
            F1500 = 1.20e-15
            F1500_err = 0.20e-15
            log_L5100 = np.log10(F1500*4*np.pi*DL**2)
            log_L5100err = np.log10(F1500_err*4*np.pi*DL**2)
            Lref = 'Barth et al. (2018)'
            c = SkyCoord.from_name('Mrk 50')
            z = 0.0234
        elif name == 'Mrk 504':
            continue # cant find BH mass
        elif name == 'Mrk 704':
            continue # cant find BH mass
        elif name == 'NGC 4593':
            mass = 6.882
            masserr = np.mean([0.084, 0.104])
            log_L5100 = 42.87
            log_L5100err = 0.18
            Lref = 'Bentz et al. (2013)'
            c = SkyCoord.from_name('NGC 4593')
            z = 0.0090
        elif name == 'PG 1310-108':
            massref = 'Williams et al. (2018)'
            DL = ((151.5*u.Mpc).to(u.cm)).value
            F1500 = 1.66e-15
            F1500_err = 0.17e-15
            mass = 6.48
            masserr = np.mean([0.21,0.18])
            log_L5100 = np.log10(F1500*4*np.pi*DL**2)
            log_L5100err = np.log10(F1500_err*4*np.pi*DL**2)
            Lref = 'Barth et al. (2018)'
            c = SkyCoord.from_name('PG 1310-108')
            z = 0.0343
        elif name == 'Zw229-015':
            mass = 6.913
            masserr = np.mean([0.075,0.119])
            DL = ((122.6*u.Mpc).to(u.cm)).value
            F1500 = 0.56e-15
            F1500_err = 0.06e-15
            log_L5100 = np.log10(F1500*4*np.pi*DL**2)
            log_L5100err = np.log10(F1500_err*4*np.pi*DL**2)
            Lref = 'Barth et al. (2018)'
            c = SkyCoord.from_name('19:05:25.9+42:27:40')
            z = 0.0279
                    
        out.append(np.array([name, c.ra.deg, c.dec.deg, mass, masserr, z, log_L5100, log_L5100err, band, unit, mjd, Vmag, Vmagerr, True,  LCref, Lref, massref]))
    return out

def gather_data_chilingarian():
    LCref = 'ZTF'
    massref = 'Chilingarian et al. (2018) [and refs within]'
    Lref = 'Chilingarian et al. (2018)' # Use Ha relation
    rm = False
    
    out = []
    
    for f in glob.glob('data/chilingarian/*.fits'):
        
        fname = os.path.basename(f.split('.')[0]).lower()
        if fname=='1':
            mass = 36e3
            masserr = 7e3
            z = 0.033
            LHa = 1.4e39
        elif fname=='2':
            mass = 65e3
            masserr = 7e3
            z = 0.037
            LHa = 3.5e39
        elif fname=='3':
            mass = 115e3
            masserr = 24e3
            z = 0.030
            LHa = 1.1e39
        elif fname=='4':
            mass = 115e3
            masserr = 38e3
            z = 0.039
            LHa = 2.3e39
        elif fname=='5':
            mass = 71e3
            masserr = 10e3
            z = 0.045
            LHa = 2.5e39
        elif fname=='7':
            continue # SKIP, in RGG sample
            mass = 111e3
            masserr = 7e3
            z = 0.039
            LHa = 6.2e39
        elif fname=='8':
            continue # SKIP, in RGG sample
            mass = 116e3
            masserr = 11e3
            z = 0.032
            LHa = 2.3e39
        elif fname=='10':
            mass=202e3
            masserr=13e3
            z = 0.072
            LHa = 21e39
        else:
            continue
            
        masserr = 0.434*masserr/mass
        mass = np.log10(mass)
        
        L5100 = 1.1934e7*LHa**(1000/1157)
        
        log_L5100 = np.log10(L5100)
        log_L5100err = 0
            
        # Load data
        data = fits.open(f)[1].data
        mjd_r_ztf = data['mjd'][data['filtercode']=='zr']
        mjd_g_ztf = data['mjd'][data['filtercode']=='zg']

        r_ztf = data['mag'][data['filtercode']=='zr']
        g_ztf = data['mag'][data['filtercode']=='zg']

        r_err_ztf = data['magerr'][data['filtercode']=='zr']
        g_err_ztf = data['magerr'][data['filtercode']=='zg']
        
        band = 'r'
        unit = 'mag'
        
        ra = data['RA'][0]
        dec = data['DEC'][0]
        
        out.append(np.array([fname, ra, dec, mass, masserr, z, log_L5100, 0, band, unit, mjd_r_ztf, r_ztf, r_err_ztf, True,  LCref, Lref, massref]))
        
    return out

def load_everything():

    # Read Kapsi
    filepaths = glob.glob("data/kapsi99/*.dat")
    out_kapsi = [gather_data_kapsi99(f) for f in filepaths]

    # Read Kelly
    filepaths = glob.glob("data/lightcurves/*.dat")
    out_kelly = [gather_data_kelly(f) for f in filepaths]
    out_kelly = [o for o in out_kelly if o is not None]

    # Gather light curves
    #filepaths = glob.glob("data/data_Colin_all/*.fits")
    filepaths = glob.glob("data/lightcurves_qian/*.fits")
    out_qian = [gather_data_qian(f) for f in filepaths]
    out_qian = [o for o in out_qian if o is not None]


    num_arg = 17

    # Read misc RM samples
    out_barth = gather_data_barth()
    out_du = gather_data_du()
    out_5273 = gather_data_5273().reshape((1,num_arg))
    out_kepler = gather_data_kepler().reshape((1,num_arg))
    out_kepler2 = gather_data_kepler2().reshape((1,num_arg))
    out_bentza = gather_data_bentz16a().reshape((1,num_arg))
    out_bentzb = gather_data_bentz16b().reshape((1,num_arg))
    out_fausnaugh = gather_data_fausnaugh()
    out_peterson = gather_data_peterson14().reshape((1,num_arg))
    out_lu = gather_data_lu16().reshape((1,num_arg))
    out_hu = gather_data_hu20().reshape((1,num_arg))
    out_crackett = gather_data_crackett20().reshape((1,num_arg))
    #out_grier17 =  gather_data_grier17()
    out_denney10 = gather_data_denney10()
    out_derosa = gather_data_derosa()
    out_rgg = gather_data_rgg()
    out_cann = gather_data_cann().reshape((1,num_arg))
    out_ztf_rm = gather_data_ztf_rm()
    out_pancoast = gather_data_pancoast()
    #out_chilingarian = gather_data_chilingarian()

    out_all = np.concatenate([out_kelly, out_qian, out_barth, out_du, out_5273, out_kepler, out_kepler2, out_bentza, out_bentzb, out_fausnaugh, 
                              out_peterson, out_lu, out_hu, out_kapsi, out_crackett, out_denney10, out_derosa, out_rgg,
                              out_ztf_rm, out_cann, out_pancoast])

    print(np.shape(out_all))
    return out_all