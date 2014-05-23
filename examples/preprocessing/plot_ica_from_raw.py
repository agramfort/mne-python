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
from mne.preprocessing import create_epochs_ecg
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

ica.fit(raw, picks=picks, decim=1, reject=dict(mag=4e-12, grad=4000e-13))

bad_inds, bad_kind, scores = ica.find_bads(raw, method='correlation')

ica.plot_scores(scores, exclude=bad_inds)  # inspect metrics used
ica.plot_sources(raw, bad_inds, start=0, stop=3.0)  # show time series
ica.plot_components(bad_inds, colorbar=False)  # show component sensitivites

ica.exclude += list(bad_inds)  # mark for exclusion

###############################################################################
# Apply ICA and visualize quality

raw_clean = ica.apply(raw)
ica.plot_overlay([raw, raw_clean])  # check the volume does not change

# show average ECG in ICA space
ecg_epochs = create_epochs_ecg(raw)
ica_ecg_epo = ica.sources_as_epochs(ecg_epochs)
ica_ecg_ave = ica_ecg_epo.average()

ica.plot_sources(ica_ecg_ave, exclude=bad_kind == 'ecg')

# overlay raw and clean ECG fields
ica.plot_overlay([ecg_epochs.average(), ica_ecg_ave])

###############################################################################
# To save an ICA session you can say:
# ica.save('my_ica.fif')
#
# You can later restore the session by saying:
# >>> from mne.preprocessing import read_ica
# >>> read_ica('my_ica.fif')
