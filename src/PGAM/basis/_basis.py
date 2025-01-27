from nemos.basis._basis import AdditiveBasis
from ._basis_mxin import GAMAdditiveBasisMixin

def GAMAdditiveBasis(AdditiveBasis, GAMAdditiveBasisMixin):

    def __init__(self, basis1, basis2):
        AdditiveBasis.__init__(self, basis1, basis2)
        GAMAdditiveBasisMixin.__init__(self)