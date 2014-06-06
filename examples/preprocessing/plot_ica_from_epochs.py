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
events = mne.find_events(raw, stim_channel='STI 014')
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False, picks=picks,
                    baseline=(None, 0), preload=True, reject=None)

###############################################################################
# Fit ICA model

ica = ICA(n_components=0.99, max_pca_components=None).fit(epochs)
print(ica)

###############################################################################
# Find EOG Artifacts

eog_inds, eog_scores = ica.find_bads_eog(epochs, threshold=4)

ica.plot_scores(eog_scores)
title = 'Sources related to %s artifacts'
ica.plot_sources(epochs, eog_inds, title=title % 'EOG')
ica.plot_components(eog_inds, title=title % 'EOG')

ica.exclude.extend(eog_inds)  # mark bad components

###############################################################################
# Assess component selection and unmixing quality

# check EOG
eog_evoked = create_eog_epochs(raw, picks=picks).average()  # get eog artifacts
ica.plot_sources(eog_evoked)  # plot eog sources + selection

# check ERF
ica.plot_overlay(eog_evoked)  # plot eog cleaning
ica.plot_overlay(epochs.average())  # plot remaining ERF
