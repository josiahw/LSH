# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 09:20:51 2016

@author: merlz
"""
import numpy
from matplotlib import pyplot

def loadTimings(fn):
    f = open(fn,'r')
    dsize = 0
    kdtime = 0
    mtime = 0
    ctime = 0
    kdconstruct = 0
    mconstruct = 0
    cconstruct = 0

    results = []
    for l in f.readlines():
        if 'size:' in l:
            results.append([dsize,kdtime,mtime,ctime,kdconstruct,mconstruct,cconstruct])
            dsize = float(l.split(':')[1].strip(' \r\n'))
        elif 'KD-tree built in' in l:
            kdconstruct = float(l.split(' ')[-2].strip(' \r\n'))
        elif 'KD-tree queried in' in l:
            kdtime = float(l.split(' ')[-2].strip(' \r\n'))
        elif 'Metric tree built in' in l:
            mconstruct = float(l.split(' ')[-2].strip(' \r\n'))
        elif 'Metric tree queried in' in l:
            mtime = float(l.split(' ')[-2].strip(' \r\n'))
        elif 'cover tree built in' in l:
            cconstruct = float(l.split(' ')[-2].strip(' \r\n'))
        elif 'cover tree queried in' in l:
            ctime = float(l.split(' ')[-2].strip(' \r\n'))
    results.append([dsize,kdtime,mtime,ctime,kdconstruct,mconstruct,cconstruct])
    results.pop(0)
    results = numpy.array(results)
    return results

def loadTimings2(fn):
    f = open(fn,'r')
    dsize = 0
    kdtime = 0
    mtime = 0
    ctime = 0
    kdconstruct = 0
    mconstruct = 0
    cconstruct = 0

    results = []
    for l in f.readlines()[1:]:
        pieces = l.strip(' \r\n').split(', ')
        r = [int(pieces[0]), pieces[1]]
        r = r + [float(i) for i in pieces[2:]]
    return results


pyplot.figure(figsize=(5.95, 3.35))


tdata = loadTimings('DataSetSizes.txt')[::19]
ndata = loadTimings2('DataSetSizes_new.txt')
ninds = [n[0] for n in ndata]
bftimes_sift = []
bftimes_gist = []
kdtimes_sift = []
kdtimes_gist = []
vptimes_sift = []
vptimes_gist = []
cttimes_sift = []
cttimes_gist = []
for n in ndata:
    if "Brute Force" in n[1]:
        bftimes_sift.append(n[2])
        bftimes_gist.append(n[3])
    elif "KD tree" in n[1]:
        kdtimes_sift.append(n[2])
        kdtimes_gist.append(n[3])
    if "VP tree" in n[1]:
        vptimes_sift.append(n[2])
        vptimes_gist.append(n[3])
    if "Cover tree" in n[1]:
        cttimes_sift.append(n[2])
        cttimes_gist.append(n[3])
pyplot.subplot(121)

pyplot.plot(tdata[:,0],tdata[:,1],lw=0.75,label="KD-tree",marker='v',ms=6.,mew=0.)
pyplot.plot(tdata[:,0],tdata[:,2],lw=0.75,label="Metric tree",marker='o',ms=6.,mew=0.)
pyplot.plot(tdata[:,0],tdata[:,3],lw=0.75,label="Cover tree",marker='s',ms=6.,mew=0.)
pyplot.plot(ninds,bftimes_sift,lw=0.75,label="Brute Force",marker='--',ms=6.,mew=0.)
pyplot.xlim((0,1000000))
pyplot.xticks(range(0,1000001,500000))


pyplot.xlabel("Data size",fontsize=10)
pyplot.ylabel("time(s)",fontsize=10)
pyplot.title("BigANN SIFT",fontsize=11)
pyplot.subplots_adjust(left=0.15, right=1., top=0.85, bottom=0.15)

axarr = [pyplot.gca()]
i = 0

pyplot.setp(axarr[i].get_xticklabels(), fontsize=9)
pyplot.setp(axarr[i].get_yticklabels(), fontsize=9)

#turn top and right borders off
almost_black = '#161616'
axarr[i].spines['top'].set_visible(False)
axarr[i].spines['right'].set_visible(False)

#make bottom and left nicer
axarr[i].spines['bottom'].set_linewidth(0.5)
axarr[i].spines['bottom'].set_color(almost_black)
axarr[i].spines['left'].set_linewidth(0.5)
axarr[i].spines['left'].set_color(almost_black)

axarr[i].xaxis.set_ticks_position('bottom')
axarr[i].yaxis.set_ticks_position('left')

pyplot.legend(fontsize=10,numpoints=2,ncol=4,loc=4,bbox_to_anchor=(2.35,-.225))



tdata = loadTimings('DataSetSizes2.txt')[::5]
pyplot.subplot(122)

pyplot.plot(tdata[:,0],tdata[:,1],lw=0.75,label="KD-tree",marker='v',ms=6.,mew=0.)
pyplot.plot(tdata[:,0],tdata[:,2],lw=0.75,label="Metric tree",marker='o',ms=6.,mew=0.)
pyplot.plot(tdata[:,0],tdata[:,3],lw=0.75,label="Cover tree",marker='s',ms=6.,mew=0.)
pyplot.plot(ninds,bftimes_gist,lw=0.75,label="Brute Force",marker='--',ms=6.,mew=0.)
pyplot.xlim((0,1000000))
pyplot.xticks(range(0,1000001,500000))


#pyplot.xlabel("Dataset size",fontsize=10)
pyplot.ylabel("time(s)",fontsize=10)
pyplot.title("BigANN GIST",fontsize=11)
pyplot.subplots_adjust(left=0.085, right=.95, top=0.85, bottom=0.15)

axarr = [pyplot.gca()]
i = 0

pyplot.setp(axarr[i].get_xticklabels(), fontsize=9)
pyplot.setp(axarr[i].get_yticklabels(), fontsize=9)

#turn top and right borders off
almost_black = '#161616'
axarr[i].spines['top'].set_visible(False)
axarr[i].spines['right'].set_visible(False)

#make bottom and left nicer
axarr[i].spines['bottom'].set_linewidth(0.5)
axarr[i].spines['bottom'].set_color(almost_black)
axarr[i].spines['left'].set_linewidth(0.5)
axarr[i].spines['left'].set_color(almost_black)

axarr[i].xaxis.set_ticks_position('bottom')
axarr[i].yaxis.set_ticks_position('left')

pyplot.savefig("../MyPHD/Thesis/Figures/exactsearchtimes.eps")