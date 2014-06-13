"""
===========================================================================
Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
===========================================================================

Decoding of motor imagery applied to EEG data decomposed using CSP.
Here the classifier is applied to features extracted on CSP filtered signals.

See http://en.wikipedia.org/wiki/Common_spatial_pattern and [1]

[1] Zoltan J. Koles. The quantitative extraction and topographic mapping
    of the abnormal components in the clinical EEG. Electroencephalography
    and Clinical Neurophysiology, 79(6):440--447, December 1991.

"""
# Authors: Martin Billinger <martin.billinger@tugraz.at>
#
# License: BSD (3-clause)

print(__doc__)
import numpy as np

try:
    # Python 3
    from urllib.request import urlretrieve as urlretrieve
except ImportError:
    # Python 2.7
    from urllib import urlretrieve as urlretrieve
import os

import mne
from mne import fiff
from mne.io.edf import read_raw_edf
from mne.datasets import sample
from mne.event import find_events
from mne.decoding import CSP
from mne.layouts import read_layout
import matplotlib.pyplot as plt


data_path = sample.data_path()


def get_data(subject, runs=[6, 10, 14]):
    """Return full path to data file and download if necessary."""
    datapath = ['S{s:03d}R{r:02d}.edf'.format(s=subject, r=r) for r in runs]
    for p, r in zip(datapath, runs):
        if not os.path.exists(p):
            url = ('http://www.physionet.org/physiobank/database/eegmmidb/'
                   'S{s:03d}/S{s:03d}R{r:02d}.edf').format(s=subject, r=r)
            print('downloading {f} from {u}...'.format(f=p, u=url))
            urlretrieve(url, p)
    return datapath


class LogCSP(CSP):
    """This class replaces the CSP's normalized bandpower features with
    log-bandpower features.

    Averaged band power is chi-square distributed. Taking the logarithm brings
    it closer to the normal distribution, which improves LDA classification.
    """
    def __init__(self, *args, **kwargs):
        self = CSP.__init__(self, *args, **kwargs)

    def transform(self, epochs_data, y=None):
        X = CSP.transform(self, epochs_data, y)
        # undo normalization
        X *= self.std_
        X += self.mean_
        # calculate log-bandpower features
        X = np.log(X)
        return X


###############################################################################
## Set parameters and read data

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = 1, 2
event_id = dict(hands=2, feet=3)
subject = 1
#runs = [3, 7, 11]  # motor execution: left hand vs right hand
#runs = [4, 8, 12]  # motor imagery: left hand vs right hand
#runs = [5, 9, 13]  # motor execution: hands vs feet
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_fnames = get_data(subject, runs)

# Read raw data from EDF files
raw = None
for fname in raw_fnames:
    raw0 = read_raw_edf(fname, tal_channel=-1, preload=True)
    try:
        raw.append(raw0)
    except AttributeError:
        raw = raw0

# Apply band-pass filter
# Hack: fake filter information so raw.filter won't throw an exception at us.
raw.info['lowpass'] = raw.info['sfreq'] * 0.5
raw.info['highpass'] = 0
raw.filter(7, 30, method='iir')

layout = read_layout('../../examples/decoding/EEG1005.lay')

# make sure channel names are in the same case
raw.info['ch_names'] = [chn.strip('.') for chn in raw.info['ch_names']]
#layout.names = [chn.upper() for chn in layout.names]

events = find_events(raw, output='onset', shortest_event=0,
                     stim_channel='STI 014')

#raw.info['bads'] = ['T8']  # set bad channels
picks = fiff.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                        exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=None, preload=True, add_eeg_ref=False)

# larger epochs for running classifier testing
epochs2 = mne.Epochs(raw, events, event_id, -1, 4, proj=True,
                    picks=picks, baseline=None, preload=True, add_eeg_ref=False)

labels = epochs.events[:, -1] - 2
evoked = epochs.average()


###############################################################################
# Classification with linear discrimant analysis

from sklearn.lda import LDA
from sklearn.cross_validation import LeaveOneOut, ShuffleSplit

n_components = 4  # pick some components
svc = LDA()
csp = LogCSP(n_components=n_components, reg=None)

# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
scores = []
epochs_data = epochs.get_data()
epochs2_data = epochs2.get_data()

fs = raw.info['sfreq']
winlen = int(fs * 0.5)  # running classifier: window length
winstep = int(fs * 0.1) # running classifier: window step size
time = np.arange(0, epochs2_data.shape[2] - winlen, winstep)

accs = []
for train_idx, test_idx in cv:
    y_train, y_test = labels[train_idx], labels[test_idx]

    X_train = csp.fit_transform(epochs_data[train_idx], y_train)
    X_test = csp.transform(epochs_data[test_idx])

    # fit classifier
    svc.fit(X_train, y_train)

    scores.append(svc.score(X_test, y_test))

    # running classifier: test classifier on sliding window
    acc = []
    for n in time:
        X_test = csp.transform(epochs2_data[test_idx][:, :, n : (n + winlen)])
        acc.append(svc.score(X_test, y_test))
    accs.append(acc)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

# Or use much more convenient scikit-learn cross_val_score function using
# a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
clf = Pipeline([('CSP', csp), ('SVC', svc)])
scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)
print(scores.mean())  # should match results above

plt.plot((time + winlen / 2) / fs + epochs2.tmin, np.mean(accs, 0))
plt.plot([0, 0], plt.gca().get_ylim(), 'k--')
plt.xlabel('time (s)')
plt.ylabel('classification accuracy')
plt.title('running classifier')

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)
evoked.data = csp.patterns_.T
evoked.times = np.arange(evoked.data.shape[0])
evoked.plot_topomap(times=[0, 1, 2, 3, 60, 61, 62, 63], ch_type='eeg', layout=layout,
                    colorbar=False, size=1.5)
