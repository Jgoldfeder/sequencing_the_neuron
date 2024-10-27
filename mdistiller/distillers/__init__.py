from ._base import Vanilla
from .CD import CD
from .KD import KD

distiller_dict = {
    "NONE": Vanilla,
    "KD": KD,
    "CD": CD
}
