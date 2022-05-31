##############################################################################
## Script to test the side-band normalization functionality of pyBumpHunter ##
##############################################################################


# Imports
import numpy as np
import pyBumpHunter as BH
import itertools as it
import matplotlib.pyplot as plt
import os


## Data generation and parameters

# Generate the reference background
np.random.seed(42)
bkg = np.random.exponential(scale=4., size=1_000_000)
print(f'bkg.shape={bkg.shape}')

# Compute histograms
hist_bkg, bins = np.histogram(bkg, bins=40, range=[0, 20])
hist_bkg = np.array(hist_bkg, dtype=float)
hist_bkg_norm = hist_bkg / 10.0

# Signal parameters
pos = 6
width = 1
Nsig = np.arange(0, 1600, 200)


## Initialization

# Initialize result arrays
rpos = np.empty((Nsig.size, 3))
rwidth = np.empty((Nsig.size, 3))
rsig = np.empty((Nsig.size, 3))
rllp = np.empty((Nsig.size, 3))
rgsig = np.empty((Nsig.size, 3))
rprms = np.empty((Nsig.size, 3))
rwrms = np.empty((Nsig.size, 3))
rsrms = np.empty((Nsig.size, 3))
rlrms = np.empty((Nsig.size, 3))
rgrms = np.empty((Nsig.size, 3, 2))

# Function to create folders only if they don't exists
def safe_mkdir(name):
    if not os.path.exists(name):
        os.mkdir(name, 0o755)
    return

# Create output folders
safe_mkdir('bumps')
safe_mkdir('bumps/SB')
safe_mkdir('bumps/SBw')
safe_mkdir('bumps/norm')
safe_mkdir('stat')
safe_mkdir('stat/SB')
safe_mkdir('stat/SBw')
safe_mkdir('stat/norm')


# Initialize 3 BumpHunter1D instances
bh_sb = BH.BumpHunter1D(
    rang = [0, 20],
    width_min=2,
    width_max=6,
    width_step=1,
    scan_step=1,
    bins=bins,
    npe=60_000,
    nworker=1,
    seed=666,
    use_sideband=True,
    sideband_width=None
)
bh_sbw = BH.BumpHunter1D(
    rang = [0, 20],
    width_min=2,
    width_max=6,
    width_step=1,
    scan_step=1,
    bins=bins,
    npe=60_000,
    nworker=1,
    seed=666,
    use_sideband=True,
    sideband_width=5
)
bh_norm = BH.BumpHunter1D(
    rang = [0, 20],
    width_min=2,
    width_max=6,
    width_step=1,
    scan_step=1,
    bins=bins,
    npe=60_000,
    nworker=1,
    seed=666,
    use_sideband=False
)



## Do the scans

# Loop over all Nsig
is_first = True
for i, n in enumerate(Nsig):
    print(f'####Nsig={n}')
    
    # Initialize the local lists
    lpos = []
    lwidth = []
    lsig = []
    lllp = []
    lgsig = []
        
    # Generate the signal
    np.random.seed(666)
    if n > 0:
        sig = np.random.normal(loc=pos, scale=width, size=n)
    
    # Repeat the scan 100 time
    for ii in range(100):
        print(f'##scan {ii}')
        
        # Create the data
        np.random.seed(ii + 100)
        if n > 0:
            data = np.append(
                np.random.exponential(scale=4.0, size=100_000),
                sig,
                axis=0
            )
        else:
            data = np.random.exponential(scale=4.0, size=100_000)
        if ii == 0:
            print(f'data.shape={data.shape}')
        
        # Make data histogram
        hist_data, _ = np.histogram(data, bins=bins, range=[0, 20])
        
        # Do the 3 scans scans
        bh_sb.bump_scan(hist_data, hist_bkg, is_hist=True, do_pseudo=is_first)
        bh_sbw.bump_scan(hist_data, hist_bkg, is_hist=True, do_pseudo=is_first)
        bh_norm.bump_scan(hist_data, hist_bkg_norm, is_hist=True, do_pseudo=is_first)
        if is_first:
            is_first = False
        
        # Append results
        left = [
            bins[bh_sb.min_loc_ar[0]],
            bins[bh_sbw.min_loc_ar[0]],
            bins[bh_norm.min_loc_ar[0]]
        ]
        right = [
            bins[bh_sb.min_loc_ar[0] + bh_sb.min_width_ar[0]],
            bins[bh_sbw.min_loc_ar[0] + bh_sbw.min_width_ar[0]],
            bins[bh_norm.min_loc_ar[0] + bh_norm.min_width_ar[0]]
        ]
        lpos.append([(l+r) / 2 for l, r in zip(left, right)])
        lwidth.append([r - l for l, r in zip(left, right)])
        lsig.append([bh_sb.signal_eval, bh_sbw.signal_eval, bh_norm.signal_eval])
        lllp.append([bh_sb.t_ar[0], bh_sbw.t_ar[0], bh_norm.t_ar[0]])
        lgsig.append([bh_sb.significance, bh_sbw.significance, bh_norm.significance])
        
        # Do bump and stat plots
        if ii == 0:
            bh_sb.plot_bump(
                hist_data,
                hist_bkg,
                is_hist=True,
                filename=f'bumps/SB/bump_Nsig={n}.png'
            )
            bh_sb.plot_stat(show_Pval=True, filename=f'stat/SB/BHstat_Nsig={n}.png')
            
            bh_sbw.plot_bump(
                hist_data,
                hist_bkg,
                is_hist=True,
                filename=f'bumps/SBw/bump_Nsig={n}.png'
            )
            bh_sbw.plot_stat(show_Pval=True, filename=f'stat/SBw/BHstat_Nsig={n}.png')
            
            bh_norm.plot_bump(
                hist_data,
                hist_bkg_norm,
                is_hist=True,
                filename=f'bumps/norm/bump_Nsig={n}.png'
            )
            bh_norm.plot_stat(show_Pval=True, filename=f'stat/norm/BHstat_Nsig={n}.png')
        
    # Convert lists to numpy arrays
    lpos = np.array(lpos)
    lwidth = np.array(lwidth)
    lsig = np.array(lsig)
    lllp = np.array(lllp)
    lgsig = np.array(lgsig)
    
    # Compute mean and RMS
    rprms[i,:] = lpos.std(axis=0)
    rwrms[i,:] = lwidth.std(axis=0)
    rsrms[i,:] = lsig.std(axis=0)
    rlrms[i,:] = lllp.std(axis=0)
    rgrms[i,:,0] = np.quantile(lgsig, 0.25, axis=0) # Take quartiles for global significance
    rgrms[i,:,1] = np.quantile(lgsig, 0.75, axis=0)
    
    rpos[i,:] = lpos.mean(axis=0)
    rwidth[i,:] = lwidth.mean(axis=0)
    rsig[i,:] = lsig.mean(axis=0)
    rllp[i,:] = lllp.mean(axis=0)
    rgsig[i,:] = np.median(lgsig, axis=0) # Take median for global significance
    
    del lpos
    del lwidth
    del lsig
    del lllp
    del lgsig
del bh_sb
del bh_sbw
del bh_norm


## Do global plots

safe_mkdir('results/')

# Reco position vs true Nsig
F = plt.figure(figsize=(10,6))
plt.errorbar(Nsig, rpos[:,0], yerr=rprms[:,0], fmt='ro', markersize=7, lw=2, label='excluded width = 0')
plt.errorbar(Nsig, rpos[:,1], yerr=rprms[:,1], fmt='bx', markersize=10 ,lw=2, label='excluded width = 5')
plt.errorbar(Nsig, rpos[:,2], yerr=rprms[:,2], fmt='gd', markersize=7 ,lw=2, label='no normalization')
plt.hlines(pos, Nsig[0], Nsig[-1], color='g', linestyle='dashed', lw=2, label='true')
plt.legend(fontsize=24)
plt.xlabel('Number of signal events', size=24)
plt.ylabel('Bump position', size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('results/SB_reco_pos.png', bbox_inches='tight')
plt.close(F)

# Reco width vs true Nsig
F = plt.figure(figsize=(10,6))
plt.errorbar(Nsig, rwidth[:,0], yerr=rwrms[:,0], fmt='ro', markersize=7, lw=2, label='excluded width = 0')
plt.errorbar(Nsig, rwidth[:,1], yerr=rwrms[:,1], fmt='bx', markersize=10, lw=2, label='excluded width = 5')
plt.errorbar(Nsig, rwidth[:,2], yerr=rwrms[:,2], fmt='gd', markersize=7, lw=2, label='no normalization')
plt.legend(fontsize=24)
plt.xlabel('Number of signal events', size=24)
plt.ylabel('Bump width', size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('results/SB_reco_width.png', bbox_inches='tight')
plt.close(F)

# Reco Nsig vs true Nsig
F = plt.figure(figsize=(10,6))
plt.errorbar(Nsig, rsig[:,0], yerr=rsrms[:,0], fmt='ro', markersize=7, lw=2, label='excluded width = 0')
plt.errorbar(Nsig, rsig[:,1], yerr=rsrms[:,1], fmt='bx', markersize=10, lw=2, label='excluded width = 5')
plt.errorbar(Nsig, rsig[:,2], yerr=rsrms[:,2], fmt='gd', markersize=7, lw=2, label='no normalization')
plt.plot(Nsig, Nsig, 'g--', lw=2, label='true')
plt.legend(fontsize=24)
plt.xlabel('Number of signal events (true)', size=24)
plt.ylabel('Evaluated number of signal events', size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('results/SB_reco_Nsig.png', bbox_inches='tight')
plt.close(F)

# Local negative log p-value vs true Nsig
F = plt.figure(figsize=(10,6))
plt.errorbar(Nsig, rllp[:,0], yerr=rlrms[:,0], fmt='ro', markersize=7, lw=2, label='excluded width = 0')
plt.errorbar(Nsig, rllp[:,1], yerr=rlrms[:,1], fmt='bx', markersize=10, lw=2, label='excluded width = 5')
plt.errorbar(Nsig, rllp[:,2], yerr=rlrms[:,2], fmt='gd', markersize=7, lw=2, label='no normalization')
plt.legend(fontsize=24)
plt.xlabel('Number of signal events', size=24)
plt.ylabel('test statistic', size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('results/SB_local_pval.png', bbox_inches='tight')
plt.close(F)

# Global significance vs true Nsig
F = plt.figure(figsize=(10,6))
plt.errorbar(
    Nsig,
    rgsig[:,0],
    yerr=[rgsig[:,0] - rgrms[:,0,0], rgrms[:,0,1] - rgsig[:,0]],
    fmt='ro',
    markersize=7,
    lw=2,
    label='excluded width = 0'
)
plt.errorbar(
    Nsig,
    rgsig[:,1],
    yerr=[rgsig[:,1] - rgrms[:,1,0], rgrms[:,1,1] - rgsig[:,1]],
    fmt='bx',
    markersize=10,
    lw=2,
    label='excluded width = 5'
)
plt.errorbar(
    Nsig,
    rgsig[:,2],
    yerr=[rgsig[:,2] - rgrms[:,2,0], rgrms[:,2,1] - rgsig[:,2]],
    fmt='gd',
    markersize=7,
    lw=2,
    label='no normalization'
)
plt.legend(fontsize=24)
plt.xlabel('Number of signal events', size=24)
plt.ylabel('global significance', size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('results/SB_global_sig.png', bbox_inches='tight')
plt.close(F)


## Ratio plots

# Reco position vs true Nsig
F = plt.figure(figsize=(10,6))
plt.plot(Nsig, rpos[:,0]/rpos[:,2], 'ro', markersize=7 ,lw=2, label='SB (sw=0) / norm')
plt.plot(Nsig, rpos[:,1]/rpos[:,2], 'bx', markersize=10 ,lw=2, label='SB (sw=5) / norm')
plt.hlines(1, Nsig[0], Nsig[-1], color='g', linestyle='dashed', lw=2)
plt.legend(fontsize=24)
plt.xlabel('Number of signal events', size=24)
plt.ylabel('mean position ratio', size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('results/SB_reco_pos_rat.png', bbox_inches='tight')
plt.close(F)

# Reco width vs true Nsig
F = plt.figure(figsize=(10,6))
plt.plot(Nsig, rwidth[:,0]/rwidth[:,2], 'ro', markersize=7 ,lw=2, label='SB (sw=0) / norm')
plt.plot(Nsig, rwidth[:,1]/rwidth[:,2], 'bx', markersize=10 ,lw=2, label='SB (sw=5) / norm')
plt.hlines(1, Nsig[0], Nsig[-1], color='g', linestyle='dashed', lw=2)
plt.legend(fontsize=24)
plt.xlabel('Number of signal events', size=24)
plt.ylabel('mean width ratio', size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('results/SB_reco_width_rat.png', bbox_inches='tight')
plt.close(F)

# Reco Nsig vs true Nsig
F = plt.figure(figsize=(10,6))
plt.plot(Nsig, rsig[:,0]/rsig[:,2], 'ro', markersize=7 ,lw=2, label='SB (sw=0) / norm')
plt.plot(Nsig, rsig[:,1]/rsig[:,2], 'bx', markersize=10 ,lw=2, label='SB (sw=5) / norm')
plt.hlines(1, Nsig[0], Nsig[-1], color='g', linestyle='dashed', lw=2)
plt.legend(fontsize=24)
plt.xlabel('Number of signal events', size=24)
plt.ylabel('mean Nsig reco ratio', size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('results/SB_reco_Nsig_rat.png', bbox_inches='tight')
plt.close(F)

# Local negative log p-value vs true Nsig
F = plt.figure(figsize=(10,6))
plt.plot(Nsig, rllp[:,0]/rllp[:,2], 'ro', markersize=7 ,lw=2, label='SB (sw=0) / norm')
plt.plot(Nsig, rllp[:,1]/rllp[:,2], 'bx', markersize=10 ,lw=2, label='SB (sw=5) / norm')
plt.hlines(1, Nsig[0], Nsig[-1], color='g', linestyle='dashed', lw=2)
plt.legend(fontsize=24)
plt.xlabel('Number of signal events', size=24)
plt.ylabel('mean test statistic ratio', size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('results/SB_local_pval_rat.png', bbox_inches='tight')
plt.close(F)

# Global significance vs true Nsig (remove first points because 0)
F = plt.figure(figsize=(10,6))
plt.plot(Nsig[1:], rgsig[1:,0]/rgsig[1:,2], 'ro', markersize=7 ,lw=2, label='SB (sw=0) / norm')
plt.plot(Nsig[1:], rgsig[1:,1]/rgsig[1:,2], 'bx', markersize=10 ,lw=2, label='SB (sw=5) / norm')
plt.hlines(1, Nsig[0], Nsig[-1], color='g', linestyle='dashed', lw=2)
plt.legend(fontsize=24)
plt.xlabel('Number of signal events', size=24)
plt.ylabel('mean global significance ratio', size=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.savefig('results/SB_global_sig_rat.png', bbox_inches='tight')
plt.close(F)


