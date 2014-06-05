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

# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne.io import Raw
from mne.preprocessing import ICA, create_eog_epochs
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


###############################################################################
# Fit ICA model
# for more background information visit the plot_ica_from_raw.py example

ica = ICA(n_components=0.90, n_pca_components=64, max_pca_components=100,
          noise_cov=None, random_state=42)

ica.fit(epochs, decim=2)
print(ica)

###############################################################################
# Find EOG Artifacts

eog_inds, eog_scores = ica.find_bads_eog(epochs)

ica.plot_scores(eog_scores)
title = 'Sources related to %s artifacts'
ica.plot_sources(epochs, eog_inds, title=title % 'ECG')
ica.plot_topomap(epochs, eog_inds, title=title % 'ECG')

###############################################################################
# Assess component selection and unmixing quality

ica.exclude += [eog_inds]  # mark bad components

# check EOG
eog_evoked = create_eog_epochs(raw).average()  # get eog artifacts
ica.plot_sources(eog_evoked)  # plot eog sources

# check ERF
ica.plot_overlay(eog_evoked)  # plot eog cleaning
ica.plot_overlay(epochs.average())  # plot remaining ERF
