from ._base import Vanilla
# from .CD import CD, CDStudent
from .KD import KD
from .CDD import CDD

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    # "CD": CD,
    "CDD": CDD,
    # "CDStudent": CDStudent
}
