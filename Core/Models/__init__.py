from .MPANet.model.baseline import Baseline as MPANet
from .Transformer.vit_pytorch import vit_base_patch16_224_TransReID,vit_small_patch16_224_TransReID,deit_small_patch16_224_TransReID,one_layer
from .PFINet.PFINet import PFINet
from .PFINet.baseline import Baseline as RGB_to_Infrared2
from .Transformer.CMTransReID import CMTransReID
from .Transformer.CMT import CMT
__all__ = ['CMT','one_layer','CMTransReID','RGB_to_Infrared2', 'PFINet', 'MPANet', 'vit_base_patch16_224_TransReID', 'vit_small_patch16_224_TransReID', 'deit_small_patch16_224_TransReID']



