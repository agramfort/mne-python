"""
=========================================
Compute Rap-Music on evoked data
=========================================

Compute a Recursively Applied and Projected MUltiple Signal Classification
(RAP-MUSIC) on evoked dataset.

The reference for Rap-Music is:
J.C. Mosher and R.M. Leahy. 1999. Source localization using recursively
applied and projected (RAP) MUSIC. Trans. Sig. Proc. 47, 2
(February 1999), 332-340.
DOI=10.1109/78.740118 http://dx.doi.org/10.1109/78.740118
"""

# Author: Yousra Bekhti <yousra.bekhti@gmail.com>
#
# License: BSD (3-clause)

import mne

import numpy as np

from mne import io
from mne.datasets import sample
from mne.beamformer import rap_music
from mne.io.pick import pick_channels_evoked


data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fwd_fname = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
evoked_fname = data_path + '/MEG/sample/sample_audvis-ave.fif'
cov_fname = data_path + '/MEG/sample/sample_audvis-cov.fif'

# Read the evoked response and crop it
condition = 'Left visual'
evoked = mne.read_evokeds(evoked_fname, condition=condition,
                          baseline=(None, 0))
evoked.crop(tmin=-50e-3, tmax=300e-3)

picks = mne.pick_types(evoked.info, meg=True)
ch_names = np.array(evoked.info['ch_names'])

evoked = pick_channels_evoked(evoked, include=ch_names[picks])

# Read the forward solution
forward = mne.read_forward_solution(fwd_fname, surf_ori=True,
                                    force_fixed=False)

# Read noise covariance matrix and regularize it
noise_cov = mne.read_cov(cov_fname)
noise_cov = mne.cov.regularize(noise_cov, evoked.info)

stc, residual = rap_music(evoked, forward, noise_cov, n_sources=4,
                          return_residual=True)

from mne.viz import plot_sparse_source_estimates
plot_sparse_source_estimates(forward['src'], stc, fig_name="Rap-Music",
                             bgcolor=(1, 1, 1), modes=["sphere"])
