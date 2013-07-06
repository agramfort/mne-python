"""
=======================
Decoding real-time data
=======================

Supervised machine learning applied to MEG data in sensor space.
Here the classifier is updated every 5 trials and the decoding
accuracy is plotted
"""
# Authors: Mainak Jas <mainak@neuro.hut.fi>
#
# License: BSD (3-clause)

print __doc__

import mne
from mne.realtime import RtClient, RtEpochs

import numpy as np
import pylab as pl

client = RtClient('localhost')
info = client.get_measurement_info()

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

chunk_size = 5  # decode every 5 trials
tr_percent = 60  # Training %
min_trials = 10  # minimum trials after which decoding should start

# select gradiometers
picks = mne.fiff.pick_types(info, meg='grad', eeg=False, eog=True,
                            stim=True, exclude=info['bads'])

# create the real-time epochs object
rt_epochs = RtEpochs(client, event_id, tmin, tmax, chunk_size,
                     consume_epochs=False, picks=picks, decim=1,
                     reject=dict(grad=4000e-13, eog=150e-6))

# start the acquisition
rt_epochs.start()

# Decoding in sensor space using a linear SVM
n_times = len(rt_epochs.times)
from sklearn.svm import SVC

times, score = [], []

pl.ion()

epochs = np.array([ep for ep in rt_epochs])
Y = np.array([y for y in rt_epochs.events])[0:chunk_size, 2]

while True:

    # Fetch epochs and labels
    epochs = np.append(epochs, np.array([ep for ep in rt_epochs]), axis=0)
    Y = np.append(Y, np.array([y for y in rt_epochs.events])[0:chunk_size, 2])

    rt_epochs.remove_old_epochs(chunk_size)

    if np.shape(epochs)[0] >= min_trials:

        # Find number of trials in training and test set
        trnum = round(np.shape(epochs)[0]*tr_percent/100)
        tsnum = np.shape(epochs)[0] - trnum

        # Separate trial and test set
        Tr_X = np.reshape(epochs[:trnum, :, :],
                          [trnum, np.shape(epochs)[2]*np.shape(epochs)[1]])
        Ts_X = np.reshape(epochs[-tsnum:, :, :],
                          [tsnum, np.shape(epochs)[2]*np.shape(epochs)[1]])
        Tr_Y = Y[:trnum]
        Ts_Y = Y[-tsnum:]

        # Online training and testing
        clf = SVC(C=1, kernel='linear')
        clf.fit(Tr_X, Tr_Y)
        result = clf.predict(Ts_X)

        acc = sum(result == Ts_Y)/tsnum*100

        times.append(1e3 * rt_epochs.times[np.shape(epochs)[0]])
        score.append(acc)

        # Plot accuracy
        pl.clf()
        pl.plot(times[-5:], score[-5:], '+', label="Classif. score")
        pl.hold(True)
        pl.plot(times[-5:], score[-5:])
        pl.axhline(50, color='k', linestyle='--', label="Chance level")
        pl.xlabel('Times (ms)')
        pl.ylabel('Classification score (% correct)')
        pl.ylim([30, 105])
        pl.title('Real-time decoding')
        pl.show()

        pl.waitforbuttonpress(0.1)
