# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:35:01 2018

@author: nsde
"""

#%%
from tensorflow.python.keras._impl.keras.layers.core import Layer
from ddtn.transformers.transformer_layers import ST_Affine_transformer
from ddtn.transformers.transformer_layers import ST_Affine_diffio_transformer
from ddtn.transformers.transformer_layers import ST_Homografy_transformer
from ddtn.transformers.transformer_layers import ST_CPAB_transformer
from ddtn.transformers.transformer_layers import ST_TPS_transformer
from ddtn.transformers.transformer_layers import ST_Affine_transformer_batch
from ddtn.transformers.transformer_layers import ST_Affine_diffio_transformer_batch
from ddtn.transformers.transformer_layers import ST_Homografy_transformer_batch
from ddtn.transformers.transformer_layers import ST_CPAB_transformer_batch
from ddtn.transformers.transformer_layers import ST_TPS_transformer_batch


#%%
class BaseTransformerLayer(Layer):
    """ Base class for defining the transformers as keras layers. Since the 
        __init__(...), compute_output_shape(...) and build(...) methods are the 
        same for all layers, we define these in this base layer
    """
    def __init__(self, localization_net, output_size):
        self.locnet = localization_net
        self.output_size = output_size
        super(BaseTransformerLayer, self).__init__()
    
    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None, int(output_size[0]), int(output_size[1]), int(input_shape[-1]))
    
    def build(self, input_shape):
        self.locnet.build(input_shape)
        self.trainable_weights = self.locnet.trainable_weights
    
    def call(self, X, mask=None):
        raise NotImplementedError("Must override call method")
        
#%%
class SpatialAffineLayer(BaseTransformerLayer):
    """ Spatial affine transformation keras layer """
    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        output = ST_Affine_transformer(X, theta, self.output_size)
        return output

#%%
class SpatialAffineDiffioLayer(BaseTransformerLayer):
    """ Spatial affine diffio transformation keras layer """
    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        output = ST_Affine_diffio_transformer(X, theta, self.output_size)
        return output

#%%
class SpatialHomografyLayer(BaseTransformerLayer):
    """ Spatial homografy transformation keras layer """
    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        output = ST_Homografy_transformer(X, theta, self.output_size)
        return output

#%%
class SpatialCPABLayer(BaseTransformerLayer):
    """ Spatial CPAB transformation keras layer """
    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        output = ST_CPAB_transformer(X, theta, self.output_size)
        return output
    
#%%
class SpatialTPSLayer(BaseTransformerLayer):
    """ Spatial TPS transformation keras layer """
    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        output = ST_TPS_transformer(X, theta, self.output_size)
        return output

#%%
class SpatialAffineBatchLayer(BaseTransformerLayer):
    """ Spatial affine batch transformation keras layer """
    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        output = ST_Affine_transformer_batch(X, theta, self.output_size)
        return output

#%%
class SpatialAffineDiffioBatchLayer(BaseTransformerLayer):
    """ Spatial affine diffio batch transformation keras layer """
    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        output = ST_Affine_diffio_transformer_batch(X, theta, self.output_size)
        return output

#%%
class SpatialHomografyBatchLayer(BaseTransformerLayer):
    """ Spatial homografy batch transformation keras layer """
    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        output = ST_Homografy_transformer_batch(X, theta, self.output_size)
        return output

#%%
class SpatialCPABBatchLayer(BaseTransformerLayer):
    """ Spatial CPAB batch transformation keras layer """
    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        output = ST_CPAB_transformer_batch(X, theta, self.output_size)
        return output

#%%
class SpatialTPSBatchLayer(BaseTransformerLayer):
    """ Spatial TPS batch transformation keras layer """
    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        output = ST_TPS_transformer_batch(X, theta, self.output_size)
        return output
 
    
#%%
if __name__ == '__main__':
    pass