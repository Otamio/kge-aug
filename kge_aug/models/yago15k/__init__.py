__all__ = ['Transe', 'Conve', 'Tucker', 'Rotate', 'Complex', 'Distmult']

import pykeen

import kge_aug.models.yago15k.v100 as v100
import kge_aug.models.yago15k.v105 as v105


if pykeen.get_version() == "1.0.0":

    Transe = v100.transe
    Distmult = v100.distmult
    Complex = v100.complex
    Conve = v100.conve
    Rotate = v100.rotate
    Tucker = v100.tucker

else:

    Transe = v105.transe
    Distmult = v105.distmult
    Complex = v105.complex
    Conve = v105.conve
    Rotate = v105.rotate
    Tucker = v105.tucker
