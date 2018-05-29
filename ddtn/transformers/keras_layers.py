# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:35:01 2018

@author: nsde
"""

#%%
from tensorflow.python.keras._impl.keras.layers.core import Layer

from ddtn.transformers.transformer_layers import ST_Affine_transformer
from ddtn.transformers.transformer_layers import ST_Affinediffeo_transformer
from ddtn.transformers.transformer_layers import ST_Homografy_transformer
from ddtn.transformers.transformer_layers import ST_CPAB_transformer
from ddtn.transformers.transformer_layers import ST_TPS_transformer
from ddtn.transformers.transformer_layers import ST_Affine_transformer_batch
from ddtn.transformers.transformer_layers import ST_Affinediffeo_transformer_batch
from ddtn.transformers.transformer_layers import ST_Homografy_transformer_batch
from ddtn.transformers.transformer_layers import ST_CPAB_transformer_batch
from ddtn.transformers.transformer_layers import ST_TPS_transformer_batch

#%%
class BaseTransformerLayer(Layer):
    """ Base class for defining the transformers as keras layers. Since the 
        __init__(...), compute_output_shape(...) and build(...) methods are the 
        same for all layers, we define these in this base layer
        
    Arguments:
        
        localization_net: a instance of keras Sequential class, that contains the
            localization network (see construc_localization_net.py for an example).
            The network should end with a Dense() layer that have as many neurons
            as the transformation that is specified

        output_size: 2-tuple with output height and width of the transformed imgs        
    """
    def __init__(self, localization_net, output_size, **kwargs):
        self.locnet = localization_net
        self.output_size = output_size
        super(BaseTransformerLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.locnet.build(input_shape)
        self._trainable_weights = self.locnet.trainable_weights
        super(BaseTransformerLayer, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (None, int(output_size[0]), int(output_size[1]), int(input_shape[-1]))

    def get_config(self):
        config = super(BaseTransformerLayer, self).get_config()
        config['localization_net'] = self.locnet
        config['output_size'] = self.output_size
        return config
    
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
class SpatialAffineDiffeoLayer(BaseTransformerLayer):
    """ Spatial affine diffio transformation keras layer """
    def call(self, X, mask=None):
        theta = self.locnet.call(X)
        output = ST_Affinediffeo_transformer(X, theta, self.output_size)
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
        output = ST_Affinediffeo_transformer_batch(X, theta, self.output_size)
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
    from ddtn.transformers.construct_localization_net import get_loc_net
    loc_net = get_loc_net((250, 250, 1), transformer_name='affine')
    custom_layer = SpatialAffineLayer(localization_net=loc_net,
                                  output_size=(250, 250))
    custom_layer.build((250, 250, 1)) # this will set the weights
