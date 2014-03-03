'''
procdata.py
Code for processing (simulated) calcium imagaing data
'''
import numpy as np
import csv
import mpf
import scipy.optimize as optm
import pickle
import time

def loadF(filen,neurons=None,headlines=0,colstart=0):
    '''
    loadF(filen,neurons=None,headlines=0,colstart=0)
    
    Load fluor traces from a csv file organized with neurons in columns and F in rows
    returns numpy matrix with each row as an image vector
    options:
    neurons = None: array of neuron indices to return. Empty returns all. Note: 0 indexed
    headlines = 0: number of header rows to skip
    colstart = 0: in which column (0 indexed) does data start
    returns numpy array of n neurons by t time
    '''
    f = open(filen,'r')
    # skip headers
    bitstart=0
    
    for dummy in range(headlines):
        h=f.readline()
        bitstart=f.tell()
    # get length of fluorescence traces
    T=0
    for t in enumerate(f):
        T+=1
    f.seek(bitstart)
    # make a numpy array to hold data
    if not(neurons):
        line1=f.readline()
        neurons=range(np.array(map(float,line1.strip().split(',')[colstart:])).size)
    
    F=np.zeros((len(neurons),T))
    
    n=0
    f.seek(bitstart)      
    # go through and pull out only columns needed
    for i in enumerate(f):
        fi=np.array(map(float,i[1].strip().split(',')[colstart:]))
        F[:,n]=fi[neurons]
        n+=1
    f.close()
    return F

def binSpikes(F,dtthr=0.05,awind=4,burstsonly=0,burstthresh=0.2):
    '''
    binSpikes(F,dtthr=0.05,awind=4)
    binarize fluorescence traces into binary spike/not spike bins
    F is numpy array of fluorescence over t time
    dtthr is the threshold for the convolved trace
    awind sample edge filter is used to find putative peaks
    considers
    returns ST is a numpy array of spike times
    and SA a numpy array containing the peak amplitude
    '''
    awind = 2*np.round(awind/2)
    # forse even though not really necessary
    Fpad=np.append(np.zeros(awind),F)
    Fpad = np.append(Fpad,np.zeros(awind))
    # awind point edge filter
    edg = np.append(np.ones(awind),-np.ones(awind+1))/(2*awind)
    edg[awind]=0
    edges = np.convolve(Fpad,edg,'valid')
    abovethr = 1.0*(edges>=dtthr)
    # get peaks of these events
    ddabovethr=abovethr[1:]-abovethr[:-1]
    ups=np.arange(len(edges))[ddabovethr>0]+1
    downs=np.arange(len(edges))[ddabovethr<0]+1
    if len(ups)>len(downs):
       downs=np.append(downs,ups[-1]) # spike at end
    if len(downs)>len(ups):
        downs=downs[1:] # spike at start
    inters=downs-ups
    spikes=np.zeros(len(ups))
    SA=np.zeros(len(ups))
    count=0
    
    for i in ups:
        spikes[count]=i-awind+np.argmax(edges[i:(i+inters[count])])
        SA[count]=np.max(edges[i:(i+inters[count])])
        count+=1
    
    ST=np.array(map(int,spikes))
    if burstsonly:
        ST=ST[SA>=burstthresh]
        SA=SA[SA>=burstthresh]
    return (ST,SA)

def sparsexcorr(ispikes,jspikes,sym=1):
    '''
    computes 0 time cross correlation between two spike trains that are sparse indexed
    inputs are the spike times of neurons i and j
    not a 'true' cross correlation since inactive times are not counted
    but for neurons which are sparsely active this is better measure
    as it only counts periods of activity and not silence
    outputs 2 scalar values i>j and j>i in tuple
    '''
    cortimes = list(set(ispikes) & set(jspikes))
    if sym:
        itoj=len(cortimes)/np.sqrt(1.0*len(ispikes))/np.sqrt(1.0*len(jspikes))
        jtoi=itoj
    else:
        itoj = len(cortimes)/np.sqrt(1.0*len(ispikes))
        jtoi = len(cortimes)/np.sqrt(1.0*len(jspikes))
    
    return (itoj,jtoi)


def writePrediction(scores,predfile,prefix,n=1000,header=['NET_neuronI_neuronJ','Strength']):
    '''
    Write out the predictions of n x n score matrix
    '''
    pred=np.zeros(n*n)
    id=[]
    count=0
    for i in range(n):
        for j in range(n):
            id.append('%s_%i_%i'%(prefix,i+1,j+1))
            pred[count]=scores[i,j]
            count+=1
    f = open(predfile,'wb')
    csvf=csv.writer(f)
    csvf.writerow(header)
    csvf.writerows(zip(id,pred))
    f.close()

def loadNetwork(netfile,n=1000,posonly=1):
    '''
    loads the connectivity network file
    creates an n x n np array of connections
    input: netfile as csv file with network connections
    n=1000, default network size
    optional: posonly=1, for negative scores, make 0
    output: links as nparray of n x n
    note: converts indices from 1 to 0 indexed
    '''
    f = open(netfile,'r')
    links = np.zeros((n,n))
    for l in enumerate(f):
        neuroni,neuronj,con=map(int,l[1].strip().split(','))
        links[neuroni-1,neuronj-1]=con
    f.close()
    if posonly:
        links=links>0
    return links

def loadPositions(posfile,n=1000):
    '''
    loads the x,y positions of each neuron
    into column vectors n x 2
    '''
    f = open(posfile,'r')
    xy = np.zeros((n,2))
    for l in enumerate(f):
        x,y=map(float,l[1].strip().split(','))
        xy[l[0],0]=x
        xy[l[0],1]=y
    f.close()
    return xy

def makeDistmat(pos):
    '''
    Make a distance matrix from x,y positions in pos
    '''
    n=pos.shape[0]
    distmat=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            d=np.sqrt((pos[i][0]-pos[j][0])**2 + (pos[i][1]-pos[j][1])**2)
            distmat[i,j]=d
            distmat[j,i]=d
    return distmat

def loadPrediction(predfile,prefix,n=1000,headlines=1):
    '''
    loads the prediction file
    outputs an n x n of floats
    '''
    f = open(predfile,'r')
    # skip the headers
    for h in range(headlines):
        f.readline()
    datastart=f.tell()
    scores = np.zeros((n,n))
    for l in enumerate(f):
        linkdir,con=l[1].strip().split(',')
        dataset,neuroni,neuronj=linkdir.split('_')
        if dataset==prefix:
            scores[int(neuroni)-1,int(neuronj)-1]=float(con)
    f.close()
    return scores


def AUC(guess,labels,n=1000):
    '''
    Compute area under the curve of receiver operator curve
    input is a file of posterior probabilities for a binary category
    note: can also be any real value expressing relative confidence in classification
    and a file containing the true classes
    outputs a scalar value
    '''

    scores = guess.reshape(guess.size)
    category = labels.reshape(labels.size)
    sortinds = np.argsort(scores)
    curval=scores[sortinds[0]]
    lastpos=0
    r=np.zeros(scores.size)
    for i in range(len(sortinds)):
        if curval != scores[sortinds[i]]:
            r[sortinds[lastpos:i]]= (lastpos+i-1)/2
            lastpos=i
            curval = scores[sortinds[i]]
        if i==(len(sortinds)-1):
            r[sortinds[lastpos:i]] = (lastpos+i)/2;
     
    auc = (np.sum(r[category==1]) - np.sum(category==1)*(np.sum(category==1)+1)/2) / \
    (np.sum(category<1)*np.sum(category==1))
    return auc

def catPredFiles(file1,file2,filenew):
    f1=open(file1,'r')
    f2=open(file2,'r')
    f3=open(filenew,'wb')
    csvf=csv.writer(f3)
    csv1=csv.reader(f1)
    for row in csv1:
        csvf.writerow(row)
    csv2=csv.reader(f2)
    c=0
    for row in csv2:
        if c>0:
            csvf.writerow(row)
        c+=1
    f1.close()
    f2.close()
    f3.close()

def makeSpikefile(Ffile,n,postfix='_spikes',batchsize=100,spikeparams={}):
    # compute the spikes from fluorescence data
    # pickle with file name same as Ffile and a postfix string
    # also returns the spike times as a list
    spikes=range(n)
    print('Computing spikes...')
    batches=range(0,n,batchsize)
    for b in batches:
        print('Processing batch %d of %d'%((b/batchsize)+1,n/batchsize))
        fls=loadF(Ffile,range(b,b+batchsize,1))
        for i in range(batchsize):
            spikes[b+i]=binSpikes(fls[i],**spikeparams)[0]
    spikefile=Ffile.split('.')[0] + postfix+ '.p'
    pickle.dump(spikes,open(spikefile,'wb')) # save out
    print('...Done')
    return spikes

def getPatterns(spikes,n,minneurons=2):
    # compute activity states from spike times
    # returns activity pattern as a n x number of states array
    
    # start by storing in a dict
    neuroncounter=0
    spikedict={}
    print('Getting active states')
    # this isn't very pythonic
    for s in spikes:
        for i in s:
            if i not in spikedict:
                spikedict[i]=[neuroncounter]
            else:
                spikedict[i].append(neuroncounter)
        neuroncounter+=1
    # note spikedict is unordered, but this doesn't matter
    # create a matrix of activity states
    nstates=len(spikedict)
    actmat=np.zeros((n,nstates))
    print('Total activity patterns: %d'%nstates)
    statecounter=0
    for acti in spikedict.items():
        if len(acti[1])>=minneurons:
            actmat[tuple(acti[1]),statecounter]=1
            statecounter+=1
    actmat=actmat[:,:statecounter-1]
    print('Found %d activity patterns'%statecounter)
    return actmat
    
def ccscore(Ffile,predfile,prefix,n,netfile=None,spikefile=None,spikekwarg={},xckwarg={}):
    # compute scores as cross correlations
    if spikefile:
        spikes=pickle.load(open(spikefile,'r'))
    else:
        spikes=makeSpikefile(Ffile,n,**spikekwarg)
            
    # get cross corr vals
    print('Computing cross correlations')
    xcmat = np.zeros((n,n))
    counter=range(n)[1:]
    for i in range(n):
        for j in counter:
            itoj,jtoi=sparsexcorr(spikes[i],spikes[j],**xckwarg)
            xcmat[i,j]=itoj
            xcmat[j,i]=jtoi
        if len(counter)>0:
            counter.pop(0)
    print('Writing out predictions')
    writePrediction(xcmat,predfile,prefix,n)
    # get auc
    if netfile:
        scores=loadPrediction(predfile,prefix,n)
        category = loadNetwork(netfile,n)
        auc=AUC(scores,category,n)
        print('AUC: %4f' %auc)


def ccscore_corr(Ffile,predfile,prefix,n,netfile=None,spikefile=None,burstfile=None,spikekwarg={},xckwarg={}):
    # compute scores as cross correlations correcting for burst to spike frequency
    # on input connections
    if spikefile:
        allspikes=pickle.load(open(spikefile,'r'))
    else:
        spikekwarg['spikeparams']['burstsonly']=0
        allspikes=makeSpikefile(Ffile,n,**spikekwarg)
    if burstfile:
        bursts=pickle.load(open(burstfile,'r'))
    else:
        spikekwarg['spikeparams']['burstsonly']=1
        bursts=makeSpikefile(Ffile,n,**spikekwarg)
    spikesonly=[]
    for i in range(n):
        spikesonly.append(list(set.difference(set(allspikes[i]),set(bursts[i]))))
    
    bothtimes=getPatterns(allspikes,n,minneurons=1)
    btimes=getPatterns(bursts,n,minneurons=1)
    stimes=getPatterns(spikesonly,n,minneurons=1)
    print('Computing cross correlations')
    xcmat = np.zeros((n,n))
    counter=range(n)[1:]
    for i in range(n):
        for j in counter:
            itoj,jtoi=sparsexcorr(allspikes[i],allspikes[j],**xckwarg)
            xcmat[i,j]=itoj
            xcmat[j,i]=jtoi
        if len(counter)>0:
            counter.pop(0)
    
    # scale inputs by burst to spike frequency ratio
    bfreq=btimes.mean(1).reshape(1,n)*1.0
    sfreq=stimes.mean(1).reshape(1,n)*1.0
    alpha=10.0
    xcmatcorr=xcmat*(1+(alpha*sfreq))
    print('Writing out predictions')
    writePrediction(xcmatcorr,predfile,prefix,n)
    if netfile:
        alphatest=np.arange(0,100,1.0)
        category = loadNetwork(netfile,n)
        for i in alphatest:
            auci=AUC(xcmat*(1+(i*sfreq)),category,n)
            print('AUC with alpha %2f: %4f' %(i,auci))
    



def isingscore(Ffile,predfile,prefix,n,netfile=None,batchsize=200,spikefile=None,minneurons=2):
    '''
    Compute connectivity likelihood using Ising model
    Uses the minimum probability flow objective function to fit interaction terms
    '''
    print('Ising model')
    # get spike/burst times from file or compute
    if spikefile:
        spikes=pickle.load(open(spikefile,'r'))
    else:
        spikes=makeSpikefile(Ffile,'_spikes',n)

    # compute activity states
    actmat=getPatterns(spikes,n,minneurons)
    
    # Initialize weights
    # to make consistent initial weight matrix, seed random number generator
    np.random.seed(314159)
    J0=np.random.rand(n,n)/np.sqrt(n)/100
    J0=J0.reshape(J0.size)
    # fit Ising model using MPF objective function and L-BFGS solver
    print('Fitting Ising Model')
    starttime=time.clock()
    J,fcost,fl=optm.fmin_l_bfgs_b(lambda x: mpf.MPFising(x, actmat, sym=0, allbitflip=0),J0,maxiter=500,factr=1e7,disp=1)
    # zero the diagonals and put on 0-1 interval
    weightfile=Ffile.split('.')[0] + '_weights.p'
    pickle.dump(J,open(weightfile,'wb')) # save out
    etime=(time.clock()-starttime)/60
    print('Fit took %2f minutes'%etime)
    J=J.reshape(n,n)
    J = J-J*np.identity(n)
    J = J - J.min()
    J= J/J.max()
    print('Writing out predictions')
    writePrediction(J,predfile,prefix,n)
    # get auc
    if netfile:
        scores=loadPrediction(predfile,prefix,n)
        category = loadNetwork(netfile,n)
        auc=AUC(scores,category,netfile,n)
        print('AUC: %4f' %auc)



if __name__=='__main__':
    Ffile='normal-1/fluorescence_normal-1.txt'
    Ffile2='small/fluorescence_iNet1_Size100_CC04inh.txt'
    
    netfile='normal-1/network_normal-1.txt'
    netfile2='small/network_iNet1_Size100_CC04inh.txt'
    
    prefix='normal'
    prefix2='smallcc04'
    
    n=1000
    n2=100
    
    predfile='results/ising_normal-1_all_sym.csv'
    predfile2='results/ising_smallcc04_all_sym.csv'
    
    spikefile='normal-1/fluorescence_normal-1_spikes.p'
    spikefile2='small/fluorescence_iNet1_Size100_CC04inh_spikes.p'
    
    burstfile='normal-1/fluorescence_normal-1_bursts.p'
    burstfile2='small/fluorescence_iNet1_Size100_CC04inh_bursts.p'
    
    #isingscore(Ffile,predfile,prefix,n,netfile,batchsize=200,spikefile=spikefile)
    #isingscore(Ffile2,predfile2,prefix2,n2,netfile2,batchsize=100,spikefile=spikefile2)
    #ccscore(Ffile2,'results/xc_smallcc04_all_sym.csv',prefix2,n2,netfile2,batchsize=100,spikefile=spikefile2,sym=1)
    smallinputs={'netfile':netfile2,'spikefile':spikefile2,'burstfile':burstfile2}
    normalinputs={'netfile':netfile,'spikefile':spikefile,'burstfile':burstfile}
    skw2={'batchsize':100,'postfix':'_bursts','spikeparams':{'burstsonly':0}}
    skw={'batchsize':200,'postfix':'_bursts','spikeparams':{'burstsonly':0}}
    xckw={'sym':1}
    #ccscore_corr(Ffile2,'results/xc_smallcc04_bscorr.csv',prefix2,n2,spikekwarg=skw2,xckwarg=xckw,**smallinputs)
    ccscore_corr(Ffile,'results/xc_normal-1_bscorr.csv',prefix,n,spikekwarg=skw,xckwarg=xckw,**normalinputs)