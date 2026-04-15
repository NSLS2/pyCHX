from pyCHX.chx_handlers import use_pims
from pyCHX.chx_libs import (
    db,
)

use_pims(
    db
)  # use pims for importing eiger data, register_handler 'AD_EIGER2' and 'AD_EIGER'
