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

# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import mne
from mne.io import Raw
from mne.preprocessing import ICA
from mne.preprocessing import create_ecg_epochs
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

# We pass a float value between 0 and 1 to select n_components based on the
# percentage of variance explained by the PCA components.

ica = ICA(n_components=0.90, max_pca_components=None)

###############################################################################
# Fit ICA model and identify bad sources

# ica.fit(raw, picks=picks, decim=1, reject=dict(mag=4e-12, grad=4000e-13))
# ica.save('test-ica.fif')
from mne.preprocessing import read_ica
ica = read_ica('test-ica.fif')

bad_inds, scores = ica.find_bads_ecg(raw)

ica.plot_scores(scores, exclude=bad_inds)  # inspect metrics used
ica.plot_sources(raw, bad_inds, start=0, stop=3.0)  # show time series
ica.plot_components(bad_inds, colorbar=False)  # show component sensitivites

ica.exclude += list(bad_inds)  # mark for exclusion

###############################################################################
# Apply ICA and visualize quality

ica.plot_overlay(raw)  # check the volume does not change

# show average ECG in ICA space
ecg_evoked = create_ecg_epochs(raw, picks=picks).average()

ica.plot_sources(ecg_evoked, exclude=bad_inds)

# overlay raw and clean ECG fields
ica.plot_overlay(ecg_evoked)


picks = mne.pick_types(raw.info, meg=True, eeg=True, exclude='bads')
# start_compare, stop_compare = raw.time_as_index([0., 3.])
# data, times = raw[picks, start_compare:stop_compare]
# raw_cln = ica.apply(raw, start=start_compare, stop=stop_compare)
# data_cln, _ = raw_cln[picks, start_compare:stop_compare]

# import matplotlib.pyplot as plt
#     # Restore sensor space data and keep all PCA components
# # let's now compare the date before and after cleaning.
# # first the raw data
# assert data.shape == data_cln.shape
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharex=True)
# ax1.plot(times, data.T, color='r')
# ax1.plot(times, data_cln.T, color='k')
# ax1.set_xlabel('time (s)')
# ax1.set_xlim(times[0], times[-1])
# ax1.set_xlim(times[0], times[-1])
# # now the affected channel
# ax2.plot(times, data.mean(0), color='r')
# ax2.plot(times, data_cln.mean(0), color='k')
# ax2.set_xlim(100, 106)
# ax2.set_xlabel('time (ms)')
# ax2.set_xlim(times[0], times[-1])


###############################################################################
# To save an ICA session you can say:
# ica.save('my_ica.fif')
#
# You can later restore the session by saying:
# >>> from mne.preprocessing import read_ica
# >>> read_ica('my_ica.fif')
