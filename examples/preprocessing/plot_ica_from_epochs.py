"""
================================
Compute ICA components on epochs
================================

ICA is used to decompose raw data in 49 to 50 sources.
The source matching the ECG is found automatically
and displayed. Finally, after the cleaned epochs are
compared to the uncleaned epochs, evoked ICA sources
are investigated using sensor space ERF plotting
techniques.

"""
print(__doc__)

# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.io import Raw
from mne.preprocessing.ica import ICA
from mne.datasets import sample

###############################################################################
# Setup paths and prepare epochs data

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)

picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True,
                       ecg=True, stim=False, exclude='bads')

tmin, tmax, event_id = -0.2, 0.5, 1
baseline = (None, 0)
reject = None

events = mne.find_events(raw, stim_channel='STI 014')
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False, picks=picks,
                    baseline=baseline, preload=True, reject=reject)

random_state = np.random.RandomState(42)

###############################################################################
# Setup ICA seed decompose data, then access and plot sources.
# for more background information visit the plot_ica_from_raw.py example

# fit sources from epochs or from raw (both works for epochs)
ica = ICA(n_components=0.90, n_pca_components=64, max_pca_components=100,
          noise_cov=None, random_state=random_state)

ica.fit(epochs, decim=2)
print(ica)

# plot spatial sensitivities of a few ICA components
title = 'Spatial patterns of ICA components (Magnetometers)'
source_idx = ica.get_sources(epochs).average().data.var(0)[:15]
ica.plot_components(source_idx, ch_type='mag')
plt.suptitle(title, fontsize=12)


###############################################################################
# Find Artifacts

ecg_inds, ecg_scores = ica.find_bads_ecg(ica)

some_trial = 10
title = 'Sources most similar to %s'
ica.plot_sources(epochs[some_trial], ecg_inds, title=title % 'ECG')

eog_inds, eog_scores = ica.find_bads_eog(ica)

ica.plot_sources(epochs, eog_inds, title=title % 'EOG')

###############################################################################
# Assess component selection and unmixing quality

# Add detected artifact sources to exclusion list
ica.exclude += [ecg_inds, eog_inds]

evoked = epochs.average()
ica.plot_sources(evoked)
ica.plot_overlay(evoked)
