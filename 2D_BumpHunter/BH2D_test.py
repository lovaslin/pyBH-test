# Imports
import numpy as np
import pyBumpHunter as BH
import itertools as it
import matplotlib.pyplot as plt
import os


## Data generation

# Generate the reference background
np.random.seed(42)
Nn = 150_000
Nbkg = 10_000_000
dscl = 100
bkg = np.random.exponential(scale=(4,4), size=(Nbkg,2))
fnoise = np.random.uniform(low=0,high=25, size=(Nn,2))
print(f'bkg.shape = {bkg.shape}')
print('correlations :')
print(np.corrcoef(bkg, rowvar=False))

# Histogram it in 2D
hist_fnoise, binx, biny = np.histogram2d(
    fnoise[:,0],
    fnoise[:,1],
    bins=[20,20],
    range=[[0,25],[0,25]]
)
hist_bkg, _, _ = np.histogram2d(
    bkg[:,0],
    bkg[:,1],
    bins=[20,20],
    range=[[0,25],[0,25]]
)
hist_bkg = hist_fnoise + hist_bkg
hist_bkg = hist_bkg / dscl

# Protect the 2D histogram in 1D
hist_bkg1 = hist_bkg.sum(axis=1)
hist_bkg2 = hist_bkg.sum(axis=0)


## Initialization

# Flaten the position/width 2D array in order to flaten the loop
pos = np.array([[5.,5.], [10.,10.], [15.,15.]])
width = np.array([[2.,2.], [4.,4.]])
wp = np.array([[s[0],s[1]] for s in it.product(pos, width)])
print(f'signal pos from {pos[0]} to {pos[-1]}')
print(f'signal width from {width[0]} to {width[-1]}')

# Choose the number of signal event
Nsig = np.arange(0, 1500, 100)
print(f'number of signals event : {Nsig} ({100 * Nsig / bkg.shape[0]}%)')
print(f'total number of signal to test : {len(wp) * Nsig.size}')

# Free some memory and keep only bkf histograms
del fnoise
del bkg
del bkg_cor

# Function to create folders only if they don't exists
def safe_mkdir(name):
    if not os.path.exists(name):
        os.mkdir(name, 0o755)
    return

# Make all needed folders
safe_mkdir('results/')
safe_mkdir('results/2D')
safe_mkdir('results/1D')

safe_mkdir('stat/')
safe_mkdir('stat/2D')
safe_mkdir('stat/1D')


# Initialize results variables for 2D scans
rpos = np.empty((len(wp), Nsig.size, 2))  # pos [wp, Nsig, xy]
rwidth = np.empty((len(wp), Nsig.size, 2)) # width [wp, Nsig, xy]
rsig = np.empty((len(wp), Nsig.size)) # signal eval  [wp, Nsi]
rllp = np.empty((len(wp), Nsig.size)) # t = -log(local p-value)  [wp, Nsig]
rgsig = np.empty((len(wp), Nsig.size)) # global significance  [wp, Nsig]
rprms = np.empty((len(wp), Nsig.size, 2))
rwrms = np.empty((len(wp), Nsig.size, 2))
rsrms = np.empty((len(wp), Nsig.size))
rlrms = np.empty((len(wp), Nsig.size))
rgrms = np.empty((len(wp), Nsig.size))

# Initialize results variables for 1D scans
rpos1 = np.empty((len(wp), Nsig.size, 2))  # pos [wp, Nsig, xy]
rwidth1 = np.empty((len(wp), Nsig.size, 2)) # width [wp, Nsig, xy]
rsig1 = np.empty((len(wp), Nsig.size, 2)) # signal eval  [wp, Nsig, xy]
rllp1 = np.empty((len(wp), Nsig.size, 2)) # t = -log(local p-value)  [wp, Nsig, xy]
rgsig1 = np.empty((len(wp), Nsig.size, 2)) # global significance  [wp, Nsig, xy]
rprms1 = np.empty((len(wp), Nsig.size, 2))
rwrms1 = np.empty((len(wp), Nsig.size, 2))
rsrms1 = np.empty((len(wp), Nsig.size, 2))
rlrms1 = np.empty((len(wp), Nsig.size, 2))
rgrms1 = np.empty((len(wp), Nsig.size, 2))

# Initiate 2 BumpHunter2D instances
bh_2d = BH.BumpHunter2D(
    rang=[[0,25],[0,25]],
    width_min=[2,2],
    width_max=[5,5],
    width_step=[1,1],
    scan_step=[1,1],
    bins=[binx,biny],
    npe=20000,
    nworker=1,
    seed=666
)

# Initiate 4 BumpHunter1D instances
bh_1d1 = BH.BumpHunter1D(
    rang=[0,25],
    width_min=2,
    width_max=5,
    width_step=1,
    scan_step=1,
    bins=binx,
    npe=20000,
    nworker=1,
    seed=666
)
bh_1d2 = BH.BumpHunter1D(
    rang=[0,25],
    width_min=2,
    width_max=5,
    width_step=1,
    scan_step=1,
    bins=biny,
    npe=20000,
    nworker=1,
    seed=666
)

## Scans loop

# Loop over the positions and width
is_first=True
for s in range(len(wp)):
    print(f'#####For pos={wp[s][0]} width={wp[s][1]}#####')
    
    # Loop over the signal strength
    for n in range(Nsig.size):
        print(f'##########Nsig={Nsig[n]}')
        
        # Initialize the local list
        lpos = np.empty((100, 2)) # [:, (no)cor, xy]
        lwidth = np.empty((100, 2)) # [:, (no)cor, xy]
        lsig = np.empty(100) # [:]
        lllp = np.empty(100) # [:]
        lgsig = np.empty(100) # [:]
        
        lpos1 = np.empty((100, 2)) # [:, xy]
        lwidth1 = np.empty((100, 2)) # [:, xy]
        lsig1 = np.empty((100, 2)) # [:, xy]
        lllp1 = np.empty((100, 2)) # [:, xy]
        lgsig1 = np.empty((100, 2)) # [:, xy]
        
        # Generate the signal
        np.random.seed(666)
        if Nsig[n] > 0:
            sig = np.random.normal(
                wp[s][0],
                wp[s][1],
                size=(Nsig[n], 2)
            )
            print(sig.shape)
            # Histogram it
            hist_sig, _, _ = np.histogram2d(sig[:,0], sig[:,1], bins=[binx,biny], range=[[0,25],[0,25]])
            hist_sig1 = hist_sig.sum(axis=1)
            hist_sig2 = hist_sig.sum(axis=0)
        
        # Loop 100 times (to repeat the scan)
        for i in range(100):
            # Create the data without correlations
            np.random.seed(i + 100)
            dnoise = np.random.uniform(low=0, high=25, size=(Nn//dscl, 2))
            data = np.random.exponential(scale=(4,4), size=(Nbkg//dscl,2))
            
            # Histogram it
            hist_dnoise, _, _ = np.histogram2d(
                dnoise[:,0],
                dnoise[:,1],
                bins=[binx,biny],
                range=[[0,25], [0,25]]
            )
            hist_data,_,_ = np.histogram2d(
                data[:,0],
                data[:,1],
                bins=[binx,biny],
                range=[[0,25], [0,25]]
            )
            hist_data = hist_dnoise + hist_data
            hist_data1 = hist_data.sum(axis=1)
            hist_data2 = hist_data.sum(axis=0)
            
            # Add signal if needed
            if Nsig[n] > 0:
                hist_data = hist_data + hist_sig
                hist_data1 = hist_data1 + hist_sig1
                hist_data2 = hist_data2 + hist_sig2
            if i == 0:
                print(f'data.shape={data.shape}')
            
            # Free some memory and keep histograms only
            del dnoise
            del data
            del data_cor
            
            # Run all the scans
            print(f'####scan{i}')
            bh_2d.bump_scan(hist_data, hist_bkg, is_hist=True, do_pseudo=is_first)
            bh_1d1.bump_scan(hist_data1, hist_bkg1, is_hist=True, do_pseudo=is_first)
            bh_1d2.bump_scan(hist_data2, hist_bkg2, is_hist=True, do_pseudo=is_first)
            if(is_first):
                is_first = False
            
            # Compute pos and width and fill local arrays (2D)
            pnocor = [
                (binx[bh_2d.min_loc_ar[0][0]] + binx[bh_2d.min_loc_ar[0][0] + bh_2d.min_width_ar[0][0]]) / 2,
                (biny[bh_2d.min_loc_ar[0][1]] + biny[bh_2d.min_loc_ar[0][1] + bh_2d.min_width_ar[0][1]]) / 2
            ]
            wnocor = [
                (binx[bh_2d.min_loc_ar[0][0] + bh_2d.min_width_ar[0][0]] - binx[bh_2d.min_loc_ar[0][0]]),
                (biny[bh_2d.min_loc_ar[0][1] + bh_2d.min_width_ar[0][1]] - biny[bh_2d.min_loc_ar[0][1]])
            ]
            lpos[i] = np.array([pnocor[0], pnocor[1]])
            lwidth[i] = np.array([wnocor[0], wnocor[1]])
            lsig[i] = bh_2d.signal_eval
            lllp[i] = bh_2d.t_ar[0]
            lgsig[i] = bh_2d.significance
            
            # Compute pos and width and fill local arrays (1D)
            pnocor = [
                (binx[bh_1d1.min_loc_ar[0]] + binx[bh_1d1.min_loc_ar[0] + bh_1d1.min_width_ar[0]]) / 2,
                (biny[bh_1d2.min_loc_ar[0]] + biny[bh_1d2.min_loc_ar[0] + bh_1d2.min_width_ar[0]]) / 2
            ]
            wnocor = [
                binx[bh_1d1.min_loc_ar[0] + bh_1d1.min_width_ar[0]] - binx[bh_1d1.min_loc_ar[0]],
                biny[bh_1d2.min_loc_ar[0] + bh_1d2.min_width_ar[0]] - biny[bh_1d2.min_loc_ar[0]]
            ]
            lpos1[i] = np.array([
                [pnocor[0], pnocor[1]]
            ])
            lwidth1[i] = np.array([wnocor[0], wnocor[1]])
            lsig1[i] = np.array([bh_1d1.signal_eval, bh_1d2.signal_eval])
            lllp1[i] = np.array([bh_1d1.t_ar[0], bh_1d2.t_ar[0]])
            lgsig1[i] = np.array([bh_1d1.significance, bh_1d2.significance])
            
            # Do the first stat plots
            if(i==0):
                bh_2d.plot_stat(
                    show_Pval=True,
                    filename=f'stat/2D/pos[{wp[s][0,0]}-{wp[s][0,1]}]'
                             f'width[{wp[s][1,0]}-{wp[s][1,1]}]Ns{Nsig[n]}.png'
                )
                bh_1d1.plot_stat(
                    show_Pval=True,
                    filename=f'stat/1D/pos[{wp[s][0,0]}-{wp[s][0,1]}]'
                             f'width[{wp[s][1,0]}-{wp[s][1,1]}]Ns{Nsig[n]}X.png'
                )
                bh_1d2.plot_stat(
                    show_Pval=True,
                    filename=f'stat/1D/pos[{wp[s][0,0]}-{wp[s][0,1]}]'
                             f'width[{wp[s][1,0]}-{wp[s][1,1]}]Ns{Nsig[n]}Y.png'
                )
        
        # Fill mean and rms to global result arrays
        rpos[s,n,:] = lpos.mean(axis=0)
        rprms[s,n,:] = lpos.std(axis=0)
        rwidth[s,n,:] = lwidth.mean(axis=0)
        rwrms[s,n,:] = lwidth.std(axis=0)
        rsig[s,n] = lsig.mean(axis=0)
        rsrms[s,n] = lsig.std(axis=0)
        rllp[s,n] = lllp.mean(axis=0)
        rlrms[s,n] = lllp.std(axis=0)
        rgsig[s,n] = lgsig.mean(axis=0)
        rgrms[s,n] = lgsig.std(axis=0)
        
        rpos1[s,n,:] = lpos1.mean(axis=0)
        rprms1[s,n,:] = lpos1.std(axis=0)
        rwidth1[s,n,:] = lwidth1.mean(axis=0)
        rwrms1[s,n,:] = lwidth1.std(axis=0)
        rsig1[s,n,:] = lsig1.mean(axis=0)
        rsrms1[s,n,:] = lsig1.std(axis=0)
        rllp1[s,n,:] = lllp1.mean(axis=0)
        rlrms1[s,n,:] = lllp1.std(axis=0)
        rgsig1[s,n,:] = lgsig1.mean(axis=0)
        rgrms1[s,n,:] = lgsig1.std(axis=0)
        
        # Free some memory before next iteration
        del lpos
        del lwidth
        del lsig
        del lllp
        del lgsig
        
        del lpos1
        del lwidth1
        del lsig1
        del lllp1
        del lgsig1
    print('##################################')
del bh_2d
del bh_1d1
del bh_1d2

## Do all the plots for all signals

# Loop over all positions/width and do plots
for s in range(len(wp)):
    ##Plots 2D
    # Reco position vs true Nsig
    F = plt.figure(figsize=(10,6))
    plt.errorbar(
        Nsig,
        rpos[s,:,0],
        yerr=rprms[s,:,0],
        fmt='ro',
        markersize=7,
        lw=2,
        label='x reco'
    )
    plt.hlines(wp[s][0,0], Nsig[0], Nsig[-1], color='g', linestyle='dashed', lw=2, label='x and y true')
    plt.errorbar(
        Nsig,
        rpos[s,:,1],
        yerr=rprms[s,:,1],
        fmt='bx',
        markersize=10,
        lw=2,
        label='y reco'
    )
    plt.legend(fontsize='xx-large')
    plt.xlabel('Nsig true', size='xx-large')
    plt.ylabel('x or y position', size='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.savefig(f'results/2D/pos[{wp[s][0,0]},{wp[s][0,1]}]_width[{wp[s][1,0]},{wp[s][1,1]}]_reco_pos.png', bbox_inches='tight')
    plt.close(F)
    
    # Reco width vs true Nsig
    F = plt.figure(figsize=(10,6))
    plt.errorbar(Nsig,rwidth[s,:,0], yerr=rwrms[s,:,0], fmt='ro', markersize=7, lw=2, label='x reco')
    plt.errorbar(Nsig,rwidth[s,:,1], yerr=rwrms[s,:,1], fmt='bx', markersize=10, lw=2, label='y reco')
    plt.legend(fontsize='xx-large')
    plt.xlabel('Nsig true', size='xx-large')
    plt.ylabel('x or y width', size='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.savefig(f'results/2D/pos[{wp[s][0,0]},{wp[s][0,1]}]_width[{wp[s][1,0]},{wp[s][1,1]}]_reco_width.png', bbox_inches='tight')
    plt.close(F)
    
    # Reco Nsig vs true Nsig
    F = plt.figure(figsize=(10,6))
    plt.errorbar(Nsig, rsig[s,:], yerr=rsrms[s,:], fmt='ro', markersize=7, lw=2,label='reco')
    plt.plot(Nsig, Nsig, 'g--',lw=2, label='true')
    plt.legend(fontsize='xx-large')
    plt.xlabel('Nsig true', size='xx-large')
    plt.ylabel('Nsig reco', size='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.savefig(f'results/2D/pos[{wp[s][0,0]},{wp[s][0,1]}]_width[{wp[s][1,0]},{wp[s][1,1]}]_reco_Nsig.png', bbox_inches='tight')
    plt.close(F)
    
    
    # Local negative log pvalue vs true Nsig
    F = plt.figure(figsize=(10,6))
    plt.errorbar(Nsig, rllp[s,:], yerr=rlrms[s,:], fmt='ro', markersize=7, lw=2)
    plt.xlabel('Nsig true', size='xx-large')
    plt.ylabel('-log(local p-value)', size='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.savefig(f'results/2D/pos[{wp[s][0,0]},{wp[s][0,1]}]_width[{wp[s][1,0]},{wp[s][1,1]}]_local_pval.png', bbox_inches='tight')
    plt.close(F)
    
    # Global significance vs true Nsig
    F = plt.figure(figsize=(10,6))
    plt.errorbar(Nsig,rgsig[s,:], yerr=rgrms[s,:], fmt='ro', markersize=7, lw=2)
    plt.xlabel('Nsig true', size='xx-large')
    plt.ylabel('global significance', size='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.savefig(f'results/2D/pos[{wp[s][0,0]},{wp[s][0,1]}]_width[{wp[s][1,0]},{wp[s][1,1]}]_global_sig.png', bbox_inches='tight')
    plt.close(F)
    
    
    ##Plots 1D
    # Reco position vs true Nsig
    F = plt.figure(figsize=(10,6))
    plt.errorbar(Nsig, rpos1[s,:,0], yerr=rprms1[s,:,0], fmt='ro', markersize=7, lw=2, label='x reco')
    plt.errorbar(Nsig, rpos1[s,:,1], yerr=rprms1[s,:,1], fmt='bx', markersize=10, lw=2, label='y reco')
    plt.hlines(wp[s][0,0], Nsig[0], Nsig[-1], color='g', linestyle='dashed', lw=2, label='x and y true')
    plt.legend(fontsize='xx-large')
    plt.xlabel('Nsig true', size='xx-large')
    plt.ylabel('x or y position', size='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.savefig(f'results/1D/pos[{wp[s][0,0]},{wp[s][0,1]}]_width[{wp[s][1,0]},{wp[s][1,1]}]_reco_pos.png', bbox_inches='tight')
    plt.close(F)
    
    # Reco width vs true Nsig
    F = plt.figure(figsize=(10,6))
    plt.errorbar(Nsig, rwidth1[s,:,0], yerr=rwrms1[s,:,0], fmt='ro', markersize=7, lw=2, label='x reco')
    plt.errorbar(Nsig, rwidth1[s,:,1], yerr=rwrms1[s,:,1], fmt='bx', markersize=10, lw=2, label='y reco')
    plt.legend(fontsize='xx-large')
    plt.xlabel('Nsig true', size='xx-large')
    plt.ylabel('x or y width', size='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.savefig(f'results/1D/pos[{wp[s][0,0]},{wp[s][0,1]}]_width[{wp[s][1,0]},{wp[s][1,1]}]_reco_width.png', bbox_inches='tight')
    plt.close(F)
    
    # Reco Nsig vs true Nsig
    F = plt.figure(figsize=(10,6))
    plt.errorbar(Nsig, rsig1[s,:,0], yerr=rsrms1[s,:,0], fmt='ro', markersize=7, lw=2, label='x reco')
    plt.errorbar(Nsig, rsig1[s,:,1], yerr=rsrms1[s,:,1], fmt='bx', markersize=10, lw=2, label='y reco')
    plt.plot(Nsig, Nsig, 'g--', lw=2, label='true')
    plt.legend(fontsize='xx-large')
    plt.xlabel('Nsig true', size='xx-large')
    plt.ylabel('Nsig reco', size='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.savefig(f'results/1D/pos[{wp[s][0,0]},{wp[s][0,1]}]_width[{wp[s][1,0]},{wp[s][1,1]}]_reco_Nsig.png', bbox_inches='tight')
    plt.close(F)
    
    # Local negative log p-value vs true Nsig
    F = plt.figure(figsize=(10,6))
    plt.errorbar(Nsig ,rllp1[s,:,0], yerr=rlrms1[s,:,0], fmt='ro', markersize=7, lw=2, label='x')
    plt.errorbar(Nsig, rllp1[s,:,1], yerr=rlrms1[s,:,1], fmt='bx', markersize=10, lw=2, label='y')
    plt.legend(fontsize='xx-large')
    plt.xlabel('Nsig true' ,size='xx-large')
    plt.ylabel('-ln(local p-value)', size='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.savefig(f'results/1D/pos[{wp[s][0,0]},{wp[s][0,1]}]_width[{wp[s][1,0]},{wp[s][1,1]}]_local_pval.png', bbox_inches='tight')
    plt.close(F)
    
    # Global significance vs true Nsig
    F = plt.figure(figsize=(10,6))
    plt.errorbar(Nsig, rgsig1[s,:,0], yerr=rgrms1[s,:,0], fmt='ro', markersize=7, lw=2, label='x')
    plt.errorbar(Nsig, rgsig1[s,:,1], yerr=rgrms1[s,:,1], fmt='bx', markersize=10, lw=2, label='y')
    plt.legend(fontsize='xx-large')
    plt.xlabel('Nsig true', size='xx-large')
    plt.ylabel('global significance', size='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.savefig(f'results/1D/pos[{wp[s][0,0]},{wp[s][0,1]}]_width[{wp[s][1,0]},{wp[s][1,1]}]_global_sig.png', bbox_inches='tight')
    plt.close(F)


