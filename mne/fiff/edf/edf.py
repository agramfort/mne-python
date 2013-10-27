"""Conversion tool from EDF+,BDF to FIF

"""

# Author: Teon Brooks <teon@nyu.edu>
#
# License: BSD (3-clause)

import os
import calendar
import datetime
import re

import numpy as np

from ...transforms import als_ras_trans_mm, apply_trans
from ...utils import verbose, logger
from ..raw import Raw
from ..meas_info import Info
from ..constants import FIFF
from ..kit.coreg import get_head_coord_trans


class RawEDF(Raw):
    """Raw object from EDF+,BDF file

    Parameters
    ----------
    input_fname : str
        Path to the EDF+,BDF file.

    n_eeg : int
        Number of EEG electrodes.

    stim_channel : str | int | None
        The channel name or channel index (starting at 0).
        If None, it will use the last channel in data.

    hpts : str | None
        Path to the hpts file containing electrode positions.
        If None, sensor locations are (0,0,0).

    annot : str | None
        Path of the annot file containing the triggering information for EDF+.
        Can be None for BDF only.
        If None for EDF+, it will raise an error.

    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.

    There is an assumption that the data are arrange such that EEG channels
    appear first then miscellaneous channels (EOGs, AUX, STIM).
    The stimulus channel is saved as 'STI 014'

    See Also
    --------
    mne.fiff.Raw : Documentation of attribute and methods.
    """
    @verbose
    def __init__(self, input_fname, n_eeg, stim_channel=None,
                 hpts=None, annot=None, preload=False, verbose=None):
        if not isinstance(n_eeg, int):
            ValueError('Must be an integer number.')
        logger.info('Extracting edf Parameters from %s...' % input_fname)
        input_fname = os.path.abspath(input_fname)
        self.info, self._edf_info = _get_edf_info(input_fname, n_eeg,
                                                  stim_channel, hpts)
        logger.info('Creating Raw.info structure...')

        # Raw attributes
        self.verbose = verbose
        self._preloaded = False
        self.fids = list()
        self._projector = None
        self.first_samp = 0
        self.last_samp = self._edf_info['nsamples'] - 1
        self.comp = None  # no compensation for KIT
        self.proj = False

        if preload:
            self._preloaded = preload
            logger.info('Reading raw data from %s...' % input_fname)
            self._data, _ = self._read_segment()
            assert len(self._data) == self.info['nchan']

            # Add time info
            self.first_samp, self.last_samp = 0, self._data.shape[1] - 1
            self._times = np.arange(self.first_samp, self.last_samp + 1,
                                    dtype=np.float64)
            self._times /= self.info['sfreq']
            logger.info('    Range : %d ... %d =  %9.3f ... %9.3f secs'
                        % (self.first_samp, self.last_samp,
                           float(self.first_samp) / self.info['sfreq'],
                           float(self.last_samp) / self.info['sfreq']))
        logger.info('Ready.')

    def __repr__(self):
        s = ('%r' % os.path.basename(self.info['filename']),
             "n_channels x n_times : %s x %s" % (len(self.ch_names),
                                       self.last_samp - self.first_samp + 1))
        return "<RawEDF  |  %s>" % ', '.join(s)

    def _read_segment(self, start=0, stop=None, sel=None, verbose=None,
                      projector=None):
        """Read a chunk of raw data

        Parameters
        ----------
        start : int, (optional)
            first sample to include (first is 0). If omitted, defaults to the
            first sample in data.

        stop : int, (optional)
            First sample to not include.
            If omitted, data is included to the end.

        sel : array, optional
            Indices of channels to select.

        projector : array
            SSP operator to apply to the data.

        verbose : bool, str, int, or None
            If not None, override default verbose level (see mne.verbose).

        Returns
        -------
        data : array, [channels x samples]
           the data matrix (channels x samples).

        times : array, [samples]
            returns the time values corresponding to the samples.
        """
        if sel is None:
            sel = range(self.info['nchan'])
        elif len(sel) == 1 and sel[0] == 0 and start == 0 and stop == 1:
            return (666, 666)
        if projector is not None:
            raise NotImplementedError('Currently does not handle projections.')
        if stop is None:
            stop = self.last_samp + 1
        elif stop > self.last_samp + 1:
            stop = self.last_samp + 1

        #  Initial checks
        start = int(start)
        stop = int(stop)

        if start >= stop:
            raise ValueError('No data in this range')

        logger.info('Reading %d ... %d  =  %9.3f ... %9.3f secs...' %
                    (start, stop - 1, start / float(self.info['sfreq']),
                               (stop - 1) / float(self.info['sfreq'])))
        sfreq = self.info['sfreq']
        data_size = self._edf_info['data_size']
        data_offset = self._edf_info['data_offset']

        with open(self.info['file_id'], 'rb') as fid:
            # extract data
            fid.seek(data_offset)
            nchan = self.info['nchan']
            buffer_size = (stop - start)
            pointer = start * nchan
            fid.seek(data_offset + pointer)
            chan_block = buffer_size / sfreq

            if self._edf_info['subtype'] == '24BIT':
                datas = []
                for _ in range(chan_block):
                    data = np.empty((nchan, sfreq), dtype=np.int32)
                    for chan in range(nchan):
                        chan_data = fid.read(sfreq * data_size)
                        chan_data
                        if isinstance(chan_data, str):
                            chan_data = np.fromstring(chan_data, np.uint8)
                        else:
                            chan_data = np.asarray(chan_data, np.uint8)
                        chan_data = chan_data.reshape(-1, 3)
                        chan_data = chan_data.astype(np.int32)
                        # this converts to 24-bit little endian integer
                        # # no support in numpy
                        chan_data = (chan_data[:, 0] +
                                     (chan_data[:, 1] << 8) +
                                     (chan_data[:, 2] << 16))
                        chan_data[chan_data >= (1 << 23)] -= (1 << 24)
                        data[chan, :] = chan_data
                    datas.append(data)
                data = np.hstack(datas)
                data = self._edf_info['gains'] * data
                stim = np.array(data[-1], int)
                mask = 255 * np.ones(stim.shape, int)
                stim = np.bitwise_and(stim, mask)
                data[-1] = stim
            else:
                data = np.fromfile(fid, dtype='<i2', count=buffer_size)
                data = data.reshape((buffer_size, nchan)).T
                # XXX : mess with digital_min and gains:
                # You should use cal and range keys in chs for fif files.
                data = ((data - self.info['digital_min']) * self._edf_info['gains']
                        + self.info['physical_min'])
                stim_channel = self._read_annot(self.info['annot'])
                data = np.vstack((data, stim_channel))
        data = data[sel]

        logger.info('[done]')
        times = np.arange(start, stop, dtype=float) / self.info['sfreq']

        return data, times

    def _read_annot(self, annot):
        """Reads an annotation file and converts it to a stimulus channel
        """

        stim_channel = unicode(annot, 'utf-8').split('\x14') if annot else []

        return stim_channel


def _get_edf_info(fname, n_eeg, stim_channel, hpts=None, annot=None):
    """Extracts all the information from the EDF+,BDF file.

    Parameters
    ----------
    rawfile : str
        Raw EDF+,BDF file to be read.

    n_eeg : int
        Number of EEG electrodes.

    stim_channel : str | int | None
        The channel name or channel index (starting at 0).
        If None, it will use the last channel in data.

    hpts : str | None
        Path to the hpts file containing electrode positions.
        If None, sensor locations are (0,0,0).

    annot : str | None
        Path of the annot file containing the triggering information for EDF+.
        Can be None for BDF only.
        If None for EDF+, it will raise an error.

    Returns
    -------
    info : instance of Info
        The measurement info.
    edf_info : dict
        A dict containing all the EDF+,BDF  specific parameters.
    """

    info = Info()
    info['file_id'] = fname
    # Add info for fif object
    info['meas_id'] = None
    info['projs'] = []
    info['comps'] = []
    info['bads'] = []
    info['acq_pars'], info['acq_stim'] = None, None
    info['filename'] = fname
    info['ctf_head_t'] = None
    info['dev_ctf_t'] = []
    info['filenames'] = []
    info['dig'] = None
    info['dev_head_t'] = None
    info['proj_id'] = None
    info['proj_name'] = None
    info['experimenter'] = None

    edf_info = dict()

    with open(fname, 'rb') as fid:
        assert(fid.tell() == 0)
        fid.seek(8)

        _ = fid.read(80).strip()  # subject id
        _ = fid.read(80).strip()  # recording id
        day, month, year = [int(x) for x in re.findall('(\d+)', fid.read(8))]
        hour, minute, sec = [int(x) for x in re.findall('(\d+)', fid.read(8))]
        date = datetime.datetime(year + 2000, month, day, hour, minute, sec)
        info['meas_date'] = calendar.timegm(date.utctimetuple())

        edf_info['data_offset'] = header_nbytes = int(fid.read(8))
        subtype = fid.read(44).strip()[:5]
        supported = ['EDF+C', 'EDF+D', '24BIT']
        if subtype not in supported:
            raise ValueError('Filetype must be either %s, %s, or %s'
                             % tuple(supported))
        edf_info['subtype'] = subtype

        n_records = int(fid.read(8))
        record_length = int(fid.read(8))  # in seconds
        info['nchan'] = int(fid.read(4))

        channels = range(info['nchan'])
        ch_names = [fid.read(16).strip() for _ in channels]
        _ = [fid.read(80).strip() for _ in channels]  # transducer type
        edf_info['units'] = [fid.read(8).strip() for _ in channels]
        not_stim_ch = np.where(np.array(edf_info['units']) != 'Boolean')[0]
        units = list(np.unique(edf_info['units']))
        if 'Boolean' in units:
            units.remove('Boolean')
        assert len(units) == 1
        if units[0] == 'uV':
            scale = 1e-6
        elif units[0] == 'V':
            scale = 1
        physical_min = np.array([float(fid.read(8)) for _ in channels])
        physical_max = np.array([float(fid.read(8)) for _ in channels])
        digital_min = np.array([float(fid.read(8)) for _ in channels])
        digital_max = np.array([float(fid.read(8)) for _ in channels])
        prefiltering = [fid.read(80).strip() for _ in channels]
        prefiltering = [re.findall('HP:\s(\w+);\sLP:\s(\d+)', filt)
                                   for filt in prefiltering[:-1]]
        if all(prefiltering):
            filt = prefiltering[0][0]
            if filt[0] == 'DC':
                info['highpass'] = 0
            else:
                info['highpass'] = int(filt[0])
            info['lowpass'] = int(filt[1])
        else:
            raise NotImplementedError('Channels contain different filtering.')
        n_samples_per_record = [int(fid.read(8)) for _ in channels]
        assert len(np.unique(n_samples_per_record)) == 1

        n_samples_per_record = n_samples_per_record[0]
        fid.read(32 * info['nchan'])  # reserved
        assert fid.tell() == header_nbytes
    physical_range = physical_max - physical_min
    digital_range = digital_max - digital_min
    edf_info['gains'] = np.array([physical_range / digital_range]).T
    edf_info['gains'][not_stim_ch] *= scale
    info['sfreq'] = int(n_samples_per_record / record_length)
    edf_info['nsamples'] = n_records * n_samples_per_record

    # Some keys to be consistent with FIF measurement info
    info['description'] = None
    info['buffer_size_sec'] = 10.
    info['orig_blocks'] = None
    info['orig_fid_str'] = None

    if edf_info['subtype'] == '24BIT':
        edf_info['data_size'] = 3  # 24-bit (3 byte) integers
    else:
        edf_info['data_size'] = 2  # 16-bit (2 byte) integers

    if hpts and os.path.lexists(hpts):
        fid = open(hpts, 'rb').read()
        locs = {}
        temp = re.findall('eeg\s(\w+)\s(-?\d+)\s(-?\d+)\s(-?\d+)', fid)
        temp = temp + re.findall('cardinal\s(\d+)\s(-?\d+)\s(-?\d+)\s(-?\d+)',
                                 fid)
        for loc in temp:
            coord = np.array(map(int, loc[1:]))
            coord = apply_trans(als_ras_trans_mm, coord)
            locs[loc[0].lower()] = coord
        trans = get_head_coord_trans(nasion=locs['2'], lpa=locs['1'],
                                     rpa=locs['3'])
        for loc in locs:
            locs[loc] = apply_trans(trans, locs[loc])
        info['dig'] = []

        point_dict = {}
        point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        point_dict['ident'] = FIFF.FIFFV_POINT_NASION
        point_dict['kind'] = FIFF.FIFFV_POINT_CARDINAL
        point_dict['r'] = apply_trans(trans, locs['2'])
        info['dig'].append(point_dict)

        point_dict = {}
        point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        point_dict['ident'] = FIFF.FIFFV_POINT_LPA
        point_dict['kind'] = FIFF.FIFFV_POINT_CARDINAL
        point_dict['r'] = apply_trans(trans, locs['1'])
        info['dig'].append(point_dict)

        point_dict = {}
        point_dict['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        point_dict['ident'] = FIFF.FIFFV_POINT_RPA
        point_dict['kind'] = FIFF.FIFFV_POINT_CARDINAL
        point_dict['r'] = apply_trans(trans, locs['3'])
        info['dig'].append(point_dict)

    else:
        locs = {}
    locs = [locs[ch_name.lower()] if ch_name.lower() in locs.keys()
            else (0, 0, 0) for ch_name in ch_names]
    sensor_locs = np.array(locs)

    if edf_info['subtype'] != '24BIT':
        if not os.path.lexists(annot):
            raise ValueError('Missing required annotation file.')

    # Creates a list of dicts of eeg channels for raw.info
    logger.info('Setting channel info structure...')
    info['chs'] = []
    if stim_channel == None:
        stim_channel = info['nchan'] - 1

    info['ch_names'] = ch_names
    for idx, ch_info in enumerate(zip(ch_names, sensor_locs), 1):
        ch_name, ch_loc = ch_info
        chan_info = {}
        chan_info['cal'] = 1.
        chan_info['logno'] = idx
        chan_info['scanno'] = idx
        chan_info['range'] = 1
        chan_info['unit_mul'] = 0
        chan_info['ch_name'] = ch_name
        chan_info['unit'] = FIFF.FIFF_UNIT_V
        chan_info['coord_frame'] = FIFF.FIFFV_COORD_HEAD
        chan_info['coil_type'] = FIFF.FIFFV_COIL_EEG
        chan_info['kind'] = FIFF.FIFFV_EEG_CH
        chan_info['eeg_loc'] = ch_loc
        chan_info['loc'] = np.zeros(12)
        chan_info['loc'][:3] = ch_loc
        check1 = stim_channel == ch_name
        check2 = stim_channel == idx
        stim_check = np.logical_or(check1, check2)
        # this deals with EOG, AUX channels
        if idx > n_eeg:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['kind'] = FIFF.FIFFV_MISC_CH
        if stim_check:
            chan_info['coil_type'] = FIFF.FIFFV_COIL_NONE
            chan_info['unit'] = FIFF.FIFF_UNIT_NONE
            chan_info['kind'] = FIFF.FIFFV_STIM_CH
            chan_info['ch_name'] = 'STI 014'
            info['ch_names'][idx - 1] = chan_info['ch_name']
        info['chs'].append(chan_info)

    # info['locs'] = sensor_locs
    return info, edf_info


def read_raw_edf(input_fname, n_eeg, stim_channel=None, hpts=None, annot=None,
                 preload=False, verbose=None):
    """Reader function for EDF+, BDF conversion to FIF

    Parameters
    ----------
    input_fname : str
        Path to the EDF+,BDF file.

    n_eeg : int
        Number of EEG electrodes.

    stim_channel : str | int | None
        The channel name or channel index (starting at 0).
        If None, it will use the last channel in data.

    hpts : str | None
        Path to the hpts file containing electrode positions.
        If None, sensor locations are (0,0,0).

    annot : str | None
        Path of the annot file containing the triggering information for EDF+.
        Can be None for BDF only.
        If None for EDF+, it will raise an error.

    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    preload : bool
        If True, all data are loaded at initialization.
        If False, data are not read until save.
    """
    return RawEDF(input_fname=input_fname, n_eeg=n_eeg,
                  stim_channel=stim_channel, hpts=hpts, annot=annot,
                  verbose=verbose, preload=preload)
