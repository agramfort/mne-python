import numpy as np
import matplotlib.pyplot as pl
#from scipy.fftpack import fftfreq

import logging
logger = logging.getLogger('mne')

import mne

from mne.fiff import Raw
from mne.datasets import sample
from mne.beamformer import iter_filter_epochs, tf_lcmv

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'
fname_fwd = data_path + '/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif'
subjects_dir = data_path + '/subjects'
label_name = 'Aud-lh'
fname_label = data_path + '/MEG/sample/labels/%s.label' % label_name

###############################################################################
# Read raw data, preload to allow filtering
raw = Raw(raw_fname, preload=True)
raw.info['bads'] = ['MEG 2443']  # 1 bad MEG channel

# Read forward operator
forward = mne.read_forward_solution(fname_fwd, surf_ori=True)

# Read label
label = mne.read_label(fname_label)

# Set picks
picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=False,
                            stim=False, exclude='bads')

# Read epochs
event_id, epoch_tmin, epoch_tmax = 1, -0.2, 0.5
events = mne.read_events(event_fname)[:3]  # TODO: Use all events
#events = mne.read_events(event_fname)  # TODO: Use all events

###############################################################################
# Time-frequency beamforming based on LCMV

# Setting frequency bins as in Dalal et al. 2008 (high gamma was subdivided)
freq_bins = [(4, 12), (12, 30), (30, 55), (65, 299)]  # Hz
win_lengths = [0.3, 0.2, 0.15, 0.1]  # s

# Setting time windows
tmin = -0.2
tmax = 0.5
tstep = 0.2
control = (-0.2, 0.0)

stcs = []
for i, (epochs_band, freq_bin) in enumerate(iter_filter_epochs(raw, freq_bins,
                                                               events,
                                                               event_id,
                                                               epoch_tmin,
                                                               epoch_tmax,
                                                               control,
                                                               n_jobs=4,
                                                               picks=picks)):
    stc = tf_lcmv(epochs_band, forward, label=label, tmin=tmin, tmax=tmax,
                  tstep=tstep, win_length=win_lengths[i], control=control,
                  reg=0.05)
    stcs.append(stc)


# Gathering results for each time window
source_power = []
for stc in stcs:
    source_power.append(stc.data)

# Finding the source with maximum source power to plot spectrogram for that
# source
source_power = np.array(source_power)
max_index = np.unravel_index(source_power.argmax(), source_power.shape)
max_source = max_index[1]

# Preparing the time and frequency grid for plotting
time_bounds = np.arange(tmin, tmax + tstep, tstep)
freq_bounds = [freq_bins[0][0]]
freq_bounds.extend([freq_bin[1] for freq_bin in freq_bins])
time_grid, freq_grid = np.meshgrid(time_bounds, freq_bounds)

# Plotting the results
# TODO: The gap between 55 and 65 Hz should be marked on the final spectrogram
pl.pcolor(time_grid, freq_grid, source_power[:, max_source, :],
          cmap=pl.cm.jet)
ax = pl.gca()
pl.xlabel('Time window boundaries [s]')
ax.set_xticks(time_bounds)
pl.xlim(time_bounds[0], time_bounds[-1])
pl.ylabel('Frequency bin boundaries [Hz]')
pl.yscale('log')
ax.set_yticks(freq_bounds)
ax.set_yticklabels([np.round(freq, 2) for freq in freq_bounds])
pl.ylim(freq_bounds[0], freq_bounds[-1])
pl.grid(True, ls='-')
pl.colorbar()
pl.show()
