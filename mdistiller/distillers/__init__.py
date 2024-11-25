from ._base import Vanilla
# from .CD import CD, CDStudent
from .KD import KD
from .CDD import CDD
from .CDD_iterative import CDD_iter

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    # "CD": CD,
    "CDD": CDD,
    "CDD_iter": CDD_iter
    # "CDStudent": CDStudent
}
