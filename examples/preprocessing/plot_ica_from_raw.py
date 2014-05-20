"""
==================================
Compute ICA components on raw data
==================================

ICA is used to decompose raw data in 49 to 50 sources.
The source matching the ECG is found automatically
and displayed. Subsequently, the cleaned data is compared
with the uncleaned data. The last section shows how to export
the sources into a fiff file for further processing and displaying, e.g.
using mne_browse_raw.

"""
print(__doc__)

# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.io import Raw
from mne.preprocessing import ICA
from mne.datasets import sample


###############################################################################
# Setup paths and prepare raw data

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'

raw = Raw(raw_fname, preload=True)
raw.filter(1, 45, n_jobs=2)

picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                       stim=False, exclude='bads')

###############################################################################
# Setup ICA seed decompose data, then access and plot sources.

# Instead of the actual number of components here we pass a float value
# between 0 and 1 to select n_components based on the percentage of
# variance explained by the PCA components.
# Also we decide to use all PCA components before mixing back to sensor space.
# Use percentages (float) or set the total number of components kept
# (int). This allows to control the amount of additional denoising.

ica = ICA(n_components=0.95, n_pca_components=1.0, max_pca_components=None,
          random_state=42)

# decompose sources for raw data using each third sample.
reject = dict(mag=4e-12, grad=4000e-13)  # exclude nonstationary segments

ica.decompose_raw(raw, picks=picks, decim=2, reject=reject)
print(ica)

###############################################################################
# Automatically find the ECG component using correlation with ECG signal.

# As we don't have an ECG channel we use the average of all magnetometers
# To improve detection, we filter the the channel and pass

ecg = raw[mne.pick_types(raw.info, meg='mag')][0].mean(0)
ecg_scores = ica.find_sources_raw(raw, ecg, score_func='pearsonr',
                                  l_freq=3, h_freq=16)

# We use outlier detection to find the relevant components
from mne.preprocessing import find_outlier_adaptive
thresh = 5.0

ecg_inds = find_outlier_adaptive(ecg_scores, thresh=thresh)

# visualize scores, source time series and topographies
title = 'correlation with ECG'
ica.plot_scores(ecg_scores, exclude=ecg_inds, title=title)
ica.plot_sources_raw(raw, ecg_inds, start=0, stop=3.0)
ica.plot_topomap(ecg_inds, colorbar=False)

ica.exclude.extend(ecg_inds)  # mark for exclusion

###############################################################################
# Automatically find the EOG component using correlation with EOG signal.

eog_ch = 'EOG 061'
eog_scores = ica.find_sources_raw(raw, eog_ch, score_func='pearsonr',
                                  l_freq=1, h_freq=10)
eog_inds = list(find_outlier_adaptive(eog_scores, thresh=thresh))

title = 'correlation with EOG'
ica.plot_scores(eog_scores, exclude=eog_inds, title=title)
ica.plot_sources_raw(raw, eog_inds, stop=3.0)
ica.plot_topomap(eog_inds, colorbar=False)

ica.exclude += eog_inds  # mark for exclusion


###############################################################################
# Show MEG data before and after ICA cleaning.

# Restore sensor space data and keep all PCA components
raw_ica = ica.pick_sources_raw(raw, include=None, n_pca_components=1.0)

# let's now compare the date before and after cleaning.
start_compare, stop_compare = raw.time_as_index([100, 106])
data, times = raw[picks, start_compare:stop_compare]
data_clean, _ = raw_ica[picks, start_compare:stop_compare]

# first the raw data
plt.figure()
plt.plot(times, data.T, color='r')
plt.plot(times, data_clean.T, color='k')
plt.xlabel('time (s)')
plt.xlim(100, 106)
plt.show()

# now the affected channel
affected_idx = raw.ch_names.index('MEG 1531')
plt.figure()
plt.plot(times, data[affected_idx], color='r')
plt.plot(times, data_clean[affected_idx], color='k')
plt.xlim(100, 106)
plt.show()


###############################################################################
# Advanced validation: check ECG components extracted

# Export ICA as Raw object for subsequent processing steps in ICA space.

from mne.preprocessing import find_ecg_events

# find ECG events
event_id = 999
events, _, _ = find_ecg_events(raw, ch_name='MEG 1531', event_id=event_id,
                               l_freq=8, h_freq=16)

# create epochs around ECG events
ecg_epochs = mne.Epochs(raw, events=events, event_id=event_id,
                        tmin=-0.5, tmax=0.5, baseline=None, proj=False,
                        picks=picks, preload=True)

# plot evoked ECG
ica.plot_sources_evoked(ecg_epochs, exclude=ecg_inds)

# plot artifact removal
ica.plot_artifact_rejection(ecg_epochs)
plt.subplots_adjust(top=0.90)

###############################################################################
# To save an ICA session you can say:
# ica.save('my_ica.fif')
#
# You can later restore the session by saying:
# >>> from mne.preprocessing import read_ica
# >>> read_ica('my_ica.fif')
