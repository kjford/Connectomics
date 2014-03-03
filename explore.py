'''
explore connectomics data
'''
import procdata as p
import numpy as np
import pickle
from sklearn.metrics import pairwise_distances

large=0
if large:
    Ffile='normal-1/fluorescence_normal-1.txt'
    netfile='normal-1/network_normal-1.txt'
    prefix='normal'
    n=1000
    predfile='results/XC_normal-1_allspikes_sym.csv'
    spikefile='normal-1/fluorescence_normal-1_spikes.p'
    posfile='normal-1/networkPositions_normal-1.txt'
else:    
    Ffile='small/fluorescence_iNet1_Size100_CC04inh.txt'
    netfile='small/network_iNet1_Size100_CC04inh.txt'
    prefix='smallcc04'
    n=100
    predfile='results/xc_smallcc03_all_sym.csv'
    spikefile='small/fluorescence_iNet1_Size100_CC04inh_spikes.p'
    burstfile='small/fluorescence_iNet1_Size100_CC04inh_bursts.p'
    posfile='small/networkPositions_iNet1_Size100_CC04inh.txt'

def VAT(d):
    dprime=np.zeros_like(d)
    n=d.shape[0]
    dmat=d+d.max()*np.identity(n) 
    
    K=range(n)
    r1=dmat.sum(axis=1).argmin()
    dprime[0]=dmat[r1]
    I=[r1]
    K.pop(r1)
    for i in range(n)[1:]:
        subd=dmat[I]
        ri,rj=np.unravel_index(subd[:,K].argmin(),subd[:,K].shape)
        #rj=subd[:,K].argmin()
        I.append(K[rj])
        if len(K)>=1:
            K.pop(rj)
    dprime=dmat[I][:,I]        
    dprime=dprime-dprime*np.identity(n)
    return dprime,I


# load network
netw=p.loadNetwork(netfile,n)
pos=p.loadPositions(posfile,n)

xc=p.loadPrediction(predfile,prefix,n)

cvmat=netw.dot(netw.T)

distmat=pairwise_distances(netw,metric='hamming')

allspikes=pickle.load(open(spikefile,'r'))
bursts=pickle.load(open(burstfile,'r'))
spikesonly=[]
for i in range(n):
    spikesonly.append(list(set.difference(set(allspikes[i]),set(bursts[i]))))


bothtimes=p.getPatterns(allspikes,n,minneurons=1)
btimes=p.getPatterns(bursts,n,minneurons=1)
stimes=p.getPatterns(spikesonly,n,minneurons=1)
bfreq=btimes.sum(1)*1.0
sfreq=stimes.sum(1)*1.0

dprime,indord=VAT(distmat)
reord=netw[indord]
reord=reord[:,reord[-1].argsort()[::-1]]

