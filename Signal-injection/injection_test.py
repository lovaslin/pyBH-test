############################################################################
## Simple example of the usage of signal injection test with pyBumpHunter ##
############################################################################

import numpy as np
import pyBumpHunter as BH
import matplotlib.pyplot as plt


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
    npe=80_000,
    nworker=1,
    signal_exp = 1000,
    str_min=0.1,
    str_step=0.1,
    str_scale='lin',
    sigma_limit=3,
    npe_inject=200,
    seed=666
)

# Run the ignal injection
bh1.signal_inject(sig_hist, bkg_hist, is_hist=True)

# Default pyBH plot
bh1.plot_inject(filename='injection_test_raw.pdf')
''''''
# Custom plot
x = np.arange(bh1.str_min, bh1.str_min + bh1.str_step * len(bh1.sigma_ar), step=bh1.str_step)
y = np.array(bh1.sigma_ar)
F = plt.figure(figsize=(12,8))
plt.errorbar(
    x,
    y[:,0],
    yerr=[y[:,1], y[:,2]],
    marker='o',
    lw=2,
    label='median $\pm$ quartiles'
)
plt.legend(loc='upper left', fontsize='xx-large')
plt.xlabel('Signal strength ($\mu$)', size='xx-large')
plt.ylabel('Global significance ($\sigma$)', size='xx-large')
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.savefig('injection_test.pdf', bbox_inches='tight')
plt.close(F)




