__all__ = ['Transe', 'Conve', 'Tucker', 'Rotate',
           'Complex', 'Distmult']

import pykeen

import kge_aug.models.yago15k.v100 as v100


if pykeen.get_version() == "1.0.0":
    Transe = v100.transe
    Distmult = v100.distmult
    Complex = v100.complex
    Conve = v100.conve
    Rotate = v100.rotate
    Tucker = v100.tucker
else:
    Transe = None
    Distmult = None
    Complex = None
    Conve = None
    Rotate = None
    Tucker = None



