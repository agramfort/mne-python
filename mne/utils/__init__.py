# # # WARNING # # #
# This list must also be updated in doc/_templates/class.rst if it is
# changed here!
_doc_special_members = ('__contains__', '__getitem__', '__iter__', '__len__',
                        '__add__', '__sub__', '__mul__', '__div__',
                        '__neg__', '__hash__')

from .check import (check_fname, check_version, check_random_state,
                    _check_fname, _check_subject, _check_pandas_installed,
                    _check_pandas_index_arguments, _check_mayavi_version,
                    _check_event_id, _check_ch_locs, _check_compensation_grade,
                    _check_if_nan, _check_type_picks, _is_numeric, _ensure_int,
                    _check_preload, _validate_type, _check_pyface_backend)
from .config import (set_config, get_config, get_config_path, set_cache_dir,
                     set_memmap_min_size, get_subjects_dir, _get_stim_channel,
                     sys_info, _get_extra_data_path, _get_root_dir,
                     _get_call_line)
from .docs import (copy_function_doc_to_method_doc, copy_doc, linkcode_resolve,
                   open_docs, deprecated)
from .fetching import _fetch_file, _url_to_local_path
from .logging import (verbose, logger, set_log_level, set_log_file,
                      use_log_level, catch_logging, warn, filter_out_warnings,
                      ETSContext)
from .misc import (run_subprocess, _pl, _clean_names, _Counter, pformat,
                   _explain_exception, _get_argvalues, sizeof_fmt)
from .progressbar import ProgressBar
from .testing import (_memory_usage, run_tests_if_main, requires_sklearn,
                      requires_version, requires_nibabel, requires_mayavi,
                      requires_good_network, requires_mne, requires_pandas,
                      requires_h5py, traits_test, requires_pysurfer,
                      ArgvSetter, SilenceStdout, has_freesurfer, has_mne_c,
                      _TempDir, has_nibabel, _import_mlab, buggy_mkl_svd,
                      requires_numpydoc, requires_tvtk, requires_freesurfer,
                      requires_nitime, requires_fs_or_nibabel, requires_dipy,
                      requires_neuromag2ft)
from .numerics import (hashfunc, md5sum, estimate_rank, _compute_row_norms,
                       _reg_pinv, random_permutation, _reject_data_segments,
                       compute_corr, _get_inst_data, array_split_idx,
                       sum_squared, split_list, _gen_events, create_slices,
                       _time_mask, grand_average, object_diff, object_hash,
                       object_size)
from .mixin import (SizeMixin, GetEpochsMixin, _prepare_read_metadata,
                    _prepare_write_metadata)
