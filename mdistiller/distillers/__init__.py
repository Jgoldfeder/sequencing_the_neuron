from ._base import Vanilla
from .CD import CD, CDStudent
from .KD import KD

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "CD": CD,
    "CDStudent": CDStudent
}
