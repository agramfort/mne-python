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
# 1) Fit ICA model and identify bad sources

ica.fit(raw, picks=picks, decim=3, reject=dict(mag=4e-12, grad=4000e-13))

bad_inds, scores = ica.find_bads_ecg(raw, threshold=3)

ica.plot_scores(scores, exclude=bad_inds)  # inspect metrics used
ica.plot_sources(raw, bad_inds, start=0, stop=3.0)  # show time series
ica.plot_components(bad_inds, colorbar=False)  # show component sensitivites

ica.exclude += list(bad_inds)  # mark for exclusion

###############################################################################
# 3) check detectionr rate and visualize artifact rejection

ica.plot_overlay(raw)  # check the volume does not change

# show average ECG in ICA space
ecg_evoked = create_ecg_epochs(raw, picks=picks).average()

ica.plot_sources(ecg_evoked, exclude=bad_inds)

# overlay raw and clean ECG fields
ica.plot_overlay(ecg_evoked)

###############################################################################
# To save an ICA solution you can say:
# ica.save('my_ica.fif')
#
# You can later restore the session by saying:
# >>> from mne.preprocessing import read_ica
# >>> read_ica('my_ica.fif')
#
# Apply the ica to Raw, Epochs or Evoked like this:
# >>> ica.apply(epochs, copy=False)