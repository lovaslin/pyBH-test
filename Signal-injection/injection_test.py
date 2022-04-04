############################################################################
## Simple example of the usage of signal injection test with pyBumpHunter ##
############################################################################

import numpy as np
import pyBumpHunter as BH


## Data Generation

# Generate some background and signal
np.random.seed(42)
bkg = np.random.exponential(scale=8.0, size=500_000)
sig = np.random.normal(loc=20, scale=3, size=10_000)

# Make histograms
rng = [0,70]
bkg_hist, bins = np.histogram(bkg, bins=50, range=rng)
bkg_hist = np.array(bkg_hist, dtype=float)  # must convert array into floats
bkg_hist = bkg_hist / 2
sig_hist, _ = np.histogram(sig, bins=50, range=rng)


# Signal injection test

# Create a BumpHunter1D instance
bh1 = BH.BumpHunter1D(
    width_min=2,
    width_max=5,
    width_step=1,
    scan_step=1,
    bins=bins,
    rang=rng,
    npe=40_000,
    nworker=1,
    signal_exp = 1000,
    str_min=0.1,
    str_step=0.1,
    str_scale='lin',
    sigma_limit=5,
    npe_inject=200,
    seed=666
)

bh1.signal_inject(sig_hist, bkg_hist, is_hist=True)

bh1.plot_inject(filename='injection_test.pdf')



