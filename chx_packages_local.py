### This enables local import of pyCHX for testing


# from pyCHX.chx_handlers import use_dask, use_pims
from chx_handlers import use_pims

# from pyCHX.chx_libs import (
from chx_libs import (
    db,
)

# changes to current version of chx_packages.py
# added load_dask_data in generic_functions


use_pims(
    db
)  # use pims for importing eiger data, register_handler 'AD_EIGER2' and 'AD_EIGER'

# from pyCHX.chx_compress import (

# from pyCHX.chx_compress_analysis import (

# from pyCHX.chx_correlationc import Get_Pixel_Arrayc, auto_two_Arrayc, cal_g2c, get_pixelist_interp_iq

# from pyCHX.chx_correlationp import _one_time_process_errorp, auto_two_Arrayp, cal_g2p, cal_GPF, get_g2_from_ROI_GPF

# from pyCHX.chx_crosscor import CrossCorrelator2, run_para_ccorr_sym

# from pyCHX.chx_generic_functions import (

# from pyCHX.chx_olog import Attachment, LogEntry, update_olog_id, update_olog_uid, update_olog_uid_with_file

# from pyCHX.chx_outlier_detection import (

# from pyCHX.chx_specklecp import (

# from pyCH.chx_xpcs_xsvs_jupyter_V1 import(

# from pyCHX.Create_Report import (

# from pyCHX.DataGonio import qphiavg

# from pyCHX.SAXS import (

# from pyCHX.Two_Time_Correlation_Function import (

# from pyCHX.XPCS_GiSAXS import (

# from pyCHX.XPCS_SAXS import (
