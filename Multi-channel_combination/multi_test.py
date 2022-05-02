# Test for the multi-channel combination (1D)
# We compare the combination method with the single channel scans (sum of channels)

import numpy as np
import matplotlib.pyplot as plt
import pyBumpHunter as BH
import os
import sys
import getopt

# Function to safely create a folder if needed
def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path, 0o755)
    return

# Get command line argument
argv = sys.argv[1:]
opts, args = getopt.getopt(argv,"","Nchan=")
Nchan=2

# Parse command line argument value
for opt, val in opts:
    if opt == '--Nchan':
        Nchan = int(val)
    else:
        print(f"Argument '{val}' not valid ...")
print(f'Nchan = {Nchan}')

# Data generation parameter
Nbkg = [100_000 + int((ch+1) * 25_000) for ch in range(Nchan)]
Nsig = [[i for ch in range(Nchan)] for i in range(0, 1300, 150)]
rng = [0, 40]
loc = [8 for ch in range(Nchan)]
scl = [1 + (ch+1) * 0.25 for ch in range(Nchan)]

# Generation of the reference background
np.random.seed(42)
bkg = [
    np.random.exponential(scale=4.5,size=Nbkg[ch]*10)
    for ch in range(Nchan)
]

# Histograminng of reference background
bins = np.histogram_bin_edges(bkg[0], bins=50, range=rng)
bkg = [
    np.histogram(bkg[ch], bins=bins, range=rng)[0]
    for ch in range(Nchan)
]
bkg = [
    np.array(bkg[ch],dtype=float)/10.0
    for ch in range(Nchan)
]

# Create 2 BumpHunter1D instance (one for each test)
bh1 = BH.BumpHunter1D(
    rang=rng,
    width_min=1,
    width_max=5,
    width_step=1,
    scan_step=1,
    bins=bins,
    npe=60_000,
    nworker=1,
    seed=666
)

bh2 = BH.BumpHunter1D(
    rang=rng,
    width_min=1,
    width_max=5,
    width_step=1,
    scan_step=1,
    bins=bins,
    npe=60_000,
    nworker=1,
    seed=666
)

# Results to be saved
pos = np.empty((2, len(Nsig), 2)) # Pos reco (real scale)
wid = np.empty((2, len(Nsig), 2)) # Width reco (real scale)
Nsi = np.empty((2, len(Nsig), 2)) # Nsig reco
llp = np.empty((2, len(Nsig), 2)) # -ln(loc p-val)
gsi = np.empty((2, len(Nsig), 2)) # Global sig

# Create the results folders if needed
safe_mkdir('BHstat')
safe_mkdir('BHstat/glob_sig')
safe_mkdir(f'BHstat/glob_sig/Nch{Nchan}')
names = ['multi-chan', 'sum']
for n in names:
    safe_mkdir('BHstat/' + n)
    safe_mkdir('BHstat/' + n + f'/Nch{Nchan}')

# Loop over Nsig
is_first = True
for s in range(len(Nsig)):
    # Generate signal if needed
    if(Nsig[s][0] > 0):
        np.random.seed(42)
        sig = [
            np.random.normal(loc=loc[ch],scale=scl[ch],size=Nsig[s][ch])
            for ch in range(Nchan)
        ]
        
        # Histogram it
        sig = [
            np.histogram(sig[ch], bins=bins, range=rng)[0]
            for ch in range(Nchan)
        ]
    
    # Initialize result containers for this Nsig
    lpos = np.empty((2, 100))
    lwid = np.empty((2, 100))
    lNsi = np.empty((2, 100))
    lllp = np.empty((2, 100))
    lgsi = np.empty((2, 100))
    
    # Loop over 100
    for i in range(100):
        # Generate data
        np.random.seed(100+i)
        data = [
            np.random.exponential(scale=4.5, size=Nbkg[ch])
            for ch in range(Nchan)
        ]
        
        # Generate histograms
        data = [
            np.histogram(data[ch], bins=bins, range=rng)[0]
            for ch in range(Nchan)
        ]
        
        # Add signal if needed
        if(Nsig[s][0] > 0):
            data = [
                data[ch] + sig[ch]
                for ch in range(Nchan)
            ]
        
        # Run BH in multi-channel
        bh1.bump_scan(
            data,
            bkg,
            multi_chan=True,
            is_hist=True,
            do_pseudo = is_first
        )
        
        # Save results
        left = bins[bh1.min_loc_ar[0]].max()
        right = bins[bh1.min_loc_ar[0] + bh1.min_width_ar[0]].min()
        lpos[0, i] = (left + right) / 2
        lwid[0, i] =  right - left
        lNsi[0, i] = bh1.signal_eval[0] + bh1.signal_eval[1]
        lllp[0, i] = bh1.t_ar[0]
        lgsi[0, i] = bh1.significance
        
        # Save the first BHstat plot
        if i == 0:
            bh1.plot_stat(show_Pval=True, filename=f'BHstat/multi-chan/Nch{Nchan}/Nsig{Nsig[s][0]}+{Nsig[s][1]}.png')
        
        # Run BH on channels sum
        data_sum = np.sum(data, axis=0)
        bkg_sum = np.sum(bkg, axis=0)
        bh2.bump_scan(
            data_sum,
            bkg_sum,
            is_hist=True,
            do_pseudo = is_first
        )
        
        # Save results
        left = bins[bh2.min_loc_ar[0]]
        right = bins[bh2.min_loc_ar[0] + bh2.min_width_ar[0]]
        lpos[1, i] = (left + right) / 2
        lwid[1, i] =  right - left
        lNsi[1, i] = bh2.signal_eval
        lllp[1, i] = bh2.t_ar[0]
        lgsi[1, i] = bh2.significance
        
        # Save the first BHstat plot
        if i == 0:
            bh2.plot_stat(show_Pval=True, filename=f'BHstat/sum/Nch{Nchan}/Nsig{Nsig[s][0]}+{Nsig[s][1]}.png')
        
        if is_first:
            is_first = False
    
    # Plot global significance distributions
    F = plt.figure(figsize=(12, 8))
    plt.title(f'Nsig = {Nsig[s][0]}+{Nsig[s][1]}', size='xx-large')
    plt.hist([lgsi[0], lgsi[1]], bins=40, histtype='step', ls='--', lw=2, label=['comb', 'sum'])
    plt.legend(fontsize='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.xlabel('Global significance', size='xx-large')
    plt.savefig(f'BHstat/glob_sig/Nch{Nchan}/Nsig{Nsig[s][0]}+{Nsig[s][1]}.png', bbox_inches='tight')
    plt.close(F)
    
    # Compute mean/rms for this Nsig
    for i in range(2):
        pos[i, s, 0] = lpos[i].mean()
        pos[i, s, 1] = lpos[i].std()
        
        wid[i, s, 0] = lwid[i].mean()
        wid[i, s, 1] = lwid[i].std()
        
        Nsi[i, s, 0] = lNsi[i].mean()
        Nsi[i, s, 1] = lNsi[i].std()
        
        llp[i, s, 0] = lllp[i].mean()
        llp[i, s, 1] = lllp[i].std()
        
        gsi[i, s, 0] = lgsi[i].mean()
        gsi[i, s, 1] = lgsi[i].std()

# Make a combined Nsig numpy array
Nsig = np.array(Nsig)
Nsig = Nsig.sum(axis=1)


## Do all the plots

safe_mkdir('results')
safe_mkdir(f'results/Nch{Nchan}')

# Plot of reco pos VS Nsig
F = plt.figure(figsize = (10,6))
plt.errorbar(
    Nsig,
    pos[0, :, 0],
    yerr = pos[0, :, 1],
    fmt='o',
    markersize=7,
    lw=2,
    color='r',
    label='multi-channel'
)
plt.errorbar(
    Nsig,
    pos[1, :, 0],
    yerr = pos[1, :, 1],
    fmt='x',
    markersize=10,
    lw=2,
    color='b',
    label='single-channel'
)
plt.hlines(loc, Nsig[0], Nsig[-1], linestyles='dashed', colors='g', lw=2, label='true')
plt.legend(fontsize='xx-large')
plt.xlabel('Number of signal events', size='xx-large')
plt.ylabel('Bump position' ,size='xx-large')
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.savefig(f'results/Nch{Nchan}/multi_reco_pos.png', bbox_inches='tight')
plt.close(F)

# Plot of reco width VS Nsig
F = plt.figure(figsize = (10,6))
plt.errorbar(
    Nsig,
    wid[0, :, 0],
    yerr = wid[0, :, 1],
    fmt='o',
    markersize=7,
    lw=2,
    color='r',
    label='multi-channel'
)
plt.errorbar(
    Nsig,
    wid[1, :, 0],
    yerr = wid[1, :, 1],
    fmt='x',
    markersize=10,
    lw=2,
    color='b',
    label='single-channel'
)
plt.legend(fontsize='xx-large')
plt.xlabel('Number of signal events', size='xx-large')
plt.ylabel('Bump width', size='xx-large')
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.savefig(f'results/Nch{Nchan}/multi_reco_width.png', bbox_inches='tight')
plt.close(F)

# Plot of reco Nsig VS Nsig true
F = plt.figure(figsize = (10,6))
plt.errorbar(
    Nsig,
    Nsi[0, :, 0],
    yerr = Nsi[0, :, 1],
    fmt='o',
    markersize=7,
    lw=2,
    color='r',
    label='multi-channel'
)
plt.errorbar(
    Nsig,
    Nsi[1, :, 0],
    yerr = Nsi[1, :, 1],
    fmt='x',
    markersize=10,
    lw=2,
    color='b',
    label='single-channel'
)
plt.plot(Nsig, Nsig, 'g--', lw=2, label='true')
plt.legend(fontsize='xx-large')
plt.xlabel('Number of signal events (true)', size='xx-large')
plt.ylabel('Evaluated number of signal events', size='xx-large')
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.savefig(f'results/Nch{Nchan}/multi_reco_Nsig.png', bbox_inches='tight')
plt.close(F)

# Plot of -ln(local p-value) VS Nsig
F = plt.figure(figsize = (10,6))
plt.errorbar(
    Nsig,
    llp[0, :, 0],
    yerr = llp[0, :, 1],
    fmt='o',
    markersize=7,
    lw=2,
    color='r',
    label='multi-channel'
)
plt.errorbar(
    Nsig,
    llp[1, :, 0],
    yerr = llp[1, :, 1],
    fmt='x',
    markersize=10,
    lw=2,
    color='b',
    label='single-channel'
)
plt.legend(fontsize='xx-large')
plt.xlabel('Number of signal events', size='xx-large')
plt.ylabel('Test statistic', size='xx-large')
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.savefig(f'results/Nch{Nchan}/multi_local_pval.png', bbox_inches='tight')
plt.close(F)

# Plot of global significance VS Nsig
F = plt.figure(figsize = (10,6))
plt.errorbar(
    Nsig,
    gsi[0, :, 0],
    yerr = gsi[0, :, 1],
    fmt='o',
    markersize=7,
    lw=2,
    color='r',
    label='multi-channel'
)
plt.errorbar(
    Nsig,
    gsi[1, :, 0],
    yerr = gsi[1, :, 1],
    fmt='x',
    markersize=10,
    lw=2,
    color='b',
    label='single-channel'
)
plt.legend(fontsize='xx-large')
plt.xlabel('Number of signal events', size='xx-large')
plt.ylabel('Global significance', size='xx-large')
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.savefig(f'results/Nch{Nchan}/multi_global_sig.png', bbox_inches='tight')
plt.close(F)


## Do all ratio plots

# Plot of reco pos VS Nsig
F = plt.figure(figsize = (10,6))
plt.plot(
    Nsig,
    pos[0, :, 0] / pos[1, :, 0],
    'o',
    markersize=7,
    lw=2,
    color='r',
)
plt.hlines(1, Nsig[0], Nsig[-1], linestyles='dashed', colors='g', lw=2)
plt.xlabel('Nsig true', size='xx-large')
plt.ylabel('Mean pos ratio (multi/sum)' ,size='xx-large')
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.savefig(f'results/Nch{Nchan}/multi_reco_pos_rat.png', bbox_inches='tight')
plt.close(F)

# Plot of reco width VS Nsig
F = plt.figure(figsize = (10,6))
plt.plot(
    Nsig,
    wid[0, :, 0] / wid[1, :, 0],
    'o',
    markersize=7,
    lw=2,
    color='r',
)
plt.hlines(1, Nsig[0], Nsig[-1], linestyles='dashed', colors='g', lw=2)
plt.xlabel('Nsig true', size='xx-large')
plt.ylabel('Mean width ratio (multi/sum)', size='xx-large')
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.savefig(f'results/Nch{Nchan}/multi_reco_width_rat.png', bbox_inches='tight')
plt.close(F)

# Plot of reco Nsig VS Nsig true
F = plt.figure(figsize = (10,6))
plt.plot(
    Nsig,
    Nsi[0, :, 0] / Nsi[1, :, 0],
    'o',
    markersize=7,
    lw=2,
    color='r',
)
plt.hlines(1, Nsig[0], Nsig[-1], linestyles='dashed', colors='g', lw=2)
plt.xlabel('Nsig true', size='xx-large')
plt.ylabel('Mean Nsig ratio (multi/sum)', size='xx-large')
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.savefig(f'results/Nch{Nchan}/multi_reco_Nsig_rat.png', bbox_inches='tight')
plt.close(F)

# Plot of -ln(local p-value) VS Nsig
F = plt.figure(figsize = (10,6))
plt.plot(
    Nsig,
    llp[0, :, 0] / llp[1, :, 0],
    'o',
    markersize=7,
    lw=2,
    color='r',
)
plt.hlines(1, Nsig[0], Nsig[-1], linestyles='dashed', colors='g', lw=2)
plt.xlabel('Nsig true', size='xx-large')
plt.ylabel('Mean test statistic ratio (multi/sum)', size='xx-large')
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.savefig(f'results/Nch{Nchan}/multi_local_pval_rat.png', bbox_inches='tight')
plt.close(F)

# Plot of global significance VS Nsig
F = plt.figure(figsize = (10,6))
plt.plot(
    Nsig,
    gsi[0, :, 0] / gsi[1, :, 0],
    'o',
    markersize=7,
    lw=2,
    color='r',
)
plt.hlines(1, Nsig[0], Nsig[-1], linestyles='dashed', colors='g', lw=2)
plt.xlabel('Nsig true', size='xx-large')
plt.ylabel('Mean global significance ratio (multi/sum)', size='xx-large')
plt.xticks(fontsize='xx-large')
plt.yticks(fontsize='xx-large')
plt.savefig(f'results/Nch{Nchan}/multi_global_sig_rat.png', bbox_inches='tight')
plt.close(F)


