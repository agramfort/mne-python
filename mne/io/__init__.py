"""FIF module for IO with .fif files"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from .constants import FIFF
from .open import fiff_open, show_fiff, _fiff_get_fid
from .evoked import (Evoked, read_evoked, write_evoked, read_evokeds,
                     write_evokeds)
from .meas_info import read_fiducials, write_fiducials, read_info, write_info
from .pick import (pick_types, pick_channels, pick_types_evoked,
                   pick_channels_regexp, pick_channels_forward,
                   pick_types_forward, pick_channels_cov,
                   pick_channels_evoked, pick_info, _has_kit_refs)

from .proj import proj_equal, make_eeg_average_ref_proj
from .cov import read_cov, write_cov
from . import array
from . import base
from . import brainvision
from . import bti
from . import edf
from . import egi
from . import fiff
from . import kit

# for backward compatibility
from .fiff import RawFIFF
from .fiff import RawFIFF as Raw
from .base import concatenate_raws, get_chpi_positions, set_eeg_reference
