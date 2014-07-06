#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import theanets
import theano

from utils import load_mnist, plot_layers

import math
import pylab as pl
import sys, getopt
from scipy.io import loadmat
from scipy.signal import butter, lfilter

import warnings
warnings.filterwarnings("ignore")
theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'warn'

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = butter(order, [low,high], btype='bandstop')
    return(b,a)


def butter_bandpass_filter(data,lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b,a,data)
    return y

def filter_noise(XX,t1,t2,sfreq):
    #t = np.linspace(t1,t2,1.5/375,endpoint=False)
    t = np.arange(t1, t2, (1.5/375) )
    beg = np.round((t1 + 0.5) * sfreq).astype(np.int)
    end = np.round((t2 + 0.5) * sfreq).astype(np.int)
    #print beg, end

    b, a = butter_bandpass(49, 51, sfreq, order=5)
    b1, a1 = butter_bandpass(99, 101, sfreq, order=5)
    

    f0 = 50
    XXnew = XX
    count =0
    for dimX in XXnew:
        for dimY in dimX:
            y = dimY[beg:end]
            ynew = lfilter(b, a, y)
            ynew2 = lfilter(b1, a1, ynew)
            dimY[beg:end] = ynew2
	    count += 1 
    return XXnew         

def plot_freq_y(XX, yy, t1, t2, sfreq, pltnum):
    pl.figure(pltnum)
    pl.clf() 
    t = np.arange(t1, t2, (1.5/375) )
    beg = np.round((t1 + 0.5) * sfreq).astype(np.int)
    end = np.round((t2 + 0.5) * sfreq).astype(np.int)
    #print beg , end

    n = np.round((t2-t1)*375/1.5)
    #print n 
    #print int((n/2)+1)
    Ffmag0 = np.zeros(int((n/2)+1),dtype=float)
    Ffangle0 = np.zeros(int((n/2)+1),dtype=float)
    Ffmag1 = np.zeros(int((n/2)+1),dtype=float)
    Ffangle1 = np.zeros(int((n/2)+1),dtype=float)
    count = 0

    XX0 = XX[np.where(yy==0)]
    XX1 = XX[np.where(yy==1)]

    for dimX in XX0:
        for dimY in dimX:
            y = dimY[beg:end]
            f = np.fft.rfft(y)
            Ffmag0 += np.abs(f)
            Ffangle0 += np.angle(f)
            count +=1
           
    Ffmag0 = Ffmag0/count
    Ffangle0 = Ffangle0/count
   
    count = 0
    for dimX in XX1:
        for dimY in dimX:
            y = dimY[beg:end]
            f = np.fft.rfft(y)
            Ffmag1 += np.abs(f)
            Ffangle1 += np.angle(f)
            count +=1
           
    Ffmag1 = Ffmag1/count
    Ffangle1 = Ffangle1/count


    k = np.arange(n)
    frq = k/1.5
    frq = frq[range(int(n/2+1))]

    pl.subplot(2,1,1)
    pl.plot(frq,Ffmag0,'r')
    pl.plot(frq,Ffmag1,'g')
    pl.xlabel('freq (Hz)')
    pl.ylabel('Avg FFT mag')
    pl.subplot(2,1,2)
    pl.plot(frq,Ffangle0,'r')
    pl.plot(frq,Ffangle1,'g')
    pl.xlabel('freq (Hz)')
    pl.ylabel('Avg FFT phase')

def plot_freq(XX,t1,t2,sfreq,pltnum):
    pl.figure(pltnum)
    pl.clf() 
    t = np.arange(t1, t2, (1.5/375) )
    beg = np.round((t1 + 0.5) * sfreq).astype(np.int)
    end = np.round((t2 + 0.5) * sfreq).astype(np.int)
    print beg , end

    n = np.round((t2-t1)*375/1.5)
    #print n 
    #print int((n/2)+1)
    Ffmag = np.zeros(int((n/2)+1),dtype=float)
    Ffangle = np.zeros(int((n/2)+1),dtype=float)
    count = 0
    for dimX in XX:
        for dimY in dimX:
            y = dimY[beg:end]
            #n = len(y)
            #print y.shape + t.shape
            f = np.fft.rfft(y)
            Ffmag += np.abs(f)
            Ffangle += np.angle(f)
            count +=1
 
    Ffmag = Ffmag/count
    Ffangle = Ffangle/count
    k = np.arange(n)
    frq = k/1.5
    frq = frq[range(int(n/2+1))]
    pl.subplot(2,1,1)
    #print np.abs(Ff).shape + frq.shape
    pl.plot(frq,Ffmag,'r')
    pl.xlabel('freq (Hz)')
    pl.ylabel('Avg FFT mag')
    pl.subplot(2,1,2)
    pl.plot(frq,Ffangle,'g')
    pl.xlabel('freq (Hz)')
    pl.ylabel('Avg FFT phase')

    return 0;

def cross_validate(clf, tr, te):
    scores = cross_validation.cross_val_score(clf, tr, te, cv=5 ) 
    print ("Cross Validation scores: mean=%f dev=%f" % (scores.mean(),scores.std()) )

def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5):
    """Creation of the feature space:
    - restricting the time window of MEG data to [tmin, tmax]sec.
    - Concatenating the 306 timeseries of each trial in one long
      vector.
    - Normalizing each feature independently (z-scoring).
    """
    #print "Applying the desired time window."
    beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
    end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
    XX = XX[:, :, beginning:end].copy()

    print XX.shape
    #print "2D Reshaping: concatenating all 306 timeseries."
    XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])

    #print "Features Normalization."
    XX -= XX.mean(0)
    XX = np.nan_to_num(XX / XX.std(0))

    return XX


if __name__ == '__main__':
    
    resfile = ''  
 
    print "DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain"
    print
    #subjects_train = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16] # use range(1, 17) for all subjects
    subjects_train = [7]
    #print "Training on subjects", subjects_train 
    subjects_val = [5]

    # We throw away all the MEG data outside the first 0.5sec from when
    # the visual stimulus start:
    tmin = 0
    tmax = 0.7
    print "Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax)

    X_train = []
    y_train = []
    X_test = []
    ids_test = []
    X_val = []
    y_val = []

    print
    #print "Creating the trainset."
    for subject in subjects_train:
        filename = 'data/train_subject%02d.mat' % subject
        print "Loading", filename
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        yy = data['y']
        sfreq = data['sfreq']
        tmin_original = data['tmin']
        #print "Dataset summary:"
        #print "XX:", XX.shape
        #print "yy:", yy.shape
        #print "sfreq:", sfreq

        XX = filter_noise(XX, -0.5, 1.0, sfreq)

	#plot_freq_y(XX, yy, -0.5, 1.0, sfreq, 1)

        XX = create_features(XX, tmin, tmax, sfreq)
        #pl.show()

        print "XX:", XX.shape
        X_train.append(XX)
        y_train.append(yy)

    for subject in subjects_val:
        filename = 'data/train_subject%02d.mat' % subject
        print "Loading val", filename
        data = loadmat(filename, squeeze_me=True)
        XX = data['X']
        yy = data['y']
        sfreq = data['sfreq']
        tmin_original = data['tmin']

        XX = filter_noise(XX, -0.5, 1.0, sfreq)

        XX = create_features(XX, tmin, tmax, sfreq)

        X_val.append(XX)
        y_val.append(yy)

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)

    print X_train.shape, y_train.shape
    print X_val.shape, y_val.shape
    n_features = X_train.shape[1]   
    print "XX:", n_features

    f_shape = int(n_features/306)

    print "Training."
    #rng = np.random.RandomState(23455)

    e = theanets.Experiment(
        theanets.Classifier,
        cnn = 1,
        input2d = 1,
        #rng = rng, 
        input_dim=(306,f_shape),
        feature_maps=(1,10),
        filter_size=(1,f_shape),
        max_pool=(1,1), 
        layers=(1000,500,2),
        train_batches=100,
    )

    train = (X_train, y_train.astype('int32'))
    valid = (X_val, y_val.astype('int32'))
    e.run(train, valid) 
    
    pl.show()

    #resfile = "submission.csv"
    if(resfile != ''):
        print "Creating the testset."
        subjects_test = range(17, 24)
        for subject in subjects_test:
            filename = 'data/test_subject%02d.mat' % subject
            print "Loading", filename
            data = loadmat(filename, squeeze_me=True)
            XX = data['X']
            ids = data['Id']
            sfreq = data['sfreq']
            tmin_original = data['tmin']
            print "Dataset summary:"
            print "XX:", XX.shape
            print "ids:", ids.shape
            print "sfreq:", sfreq
            XX = filter_noise(XX, -0.5, 1.0, sfreq)
            XX = create_features(XX, tmin, tmax, sfreq)
            X_test.append(XX)
            ids_test.append(ids)

        X_test = np.vstack(X_test)
        ids_test = np.concatenate(ids_test)
        print "Testset:", X_test.shape
                      
        print "Predicting."
        y_pred = e.network.predict(X_test)

        print "Creating submission file", resfile
        f = open(resfile, "w")
        print >> f, "Id,Prediction"
        for i in range(len(y_pred)):
            print >> f, str(ids_test[i]) + "," + str(y_pred[i])
        f.close()
    print "Done."
    
