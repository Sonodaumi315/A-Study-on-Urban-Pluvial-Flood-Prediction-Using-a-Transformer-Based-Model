import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import ViTModel, ViTPreTrainedModel, ViTConfig
from typing import Dict, List, Optional, Set, Tuple, Union
import collections.abc

# General docstring
_CONFIG_FOR_DOC = "ViTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/vit-base-patch16-224-in21k"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/vit-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


VIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/vit-base-patch16-224",
    # See all ViT models at https://huggingface.co/models?filter=vit
]

class ViTEmbeddingsWithRain(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, num_ranin_patch: int = 1, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbeddingsWithRain(config)
        num_patches = self.patch_embeddings.num_patches
        self.num_ranin_patch = num_ranin_patch
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1 + self.num_ranin_patch, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        rain_p: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values=pixel_values, rain_p=rain_p, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings

class ViTPatchEmbeddingsWithRain(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, rain_p: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        
        embeddings = torch.cat((embeddings, rain_p), dim = 1)
        return embeddings

class ViTEPConfig(ViTConfig):
    model_type = "vitep"

    def __init__(
        self,
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="gelu",
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=256,
        patch_size=16,
        num_channels=6,
        qkv_bias=True,
        encoder_stride=16,
        num_rain_patch=1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride
        self.num_rain_patch = num_rain_patch

class ViTEP(ViTPreTrainedModel):
    def __init__(self, config = ViTEPConfig()):
        super().__init__(config)
        self.config           = config
        self.hidden_size      = config.hidden_size
        self.image_size       = config.image_size #config["image_size"]
        self.patch_size       = config.patch_size #config["patch_size"]
        self.num_rain_patch   = config.num_rain_patch
        
        self.model            = ViTModel(config, add_pooling_layer=False)
        self.model.embeddings = ViTEmbeddingsWithRain(config, self.num_rain_patch, use_mask_token=False)
        
        self.linear     = nn.Linear(self.hidden_size, self.patch_size*self.patch_size)
        self.MSELoss    = nn.MSELoss()
        self.patch_num  = int((self.image_size/self.patch_size)**2)
        
        self.rain_vec     = nn.Linear(9, self.hidden_size*self.num_rain_patch)
        self.leakyrelu    = nn.LeakyReLU(negative_slope = 0.2, inplace=True)
    
    def vec2img(self, results):
        batch_size = results.shape[0]
        
        sf = nn.PixelShuffle(16)
        ans = results.transpose(1, 2).reshape(batch_size, 256, 16, 16)
        ans = sf(ans)#.transpose(2, 3).reshape(batch_size, self.image_size, self.image_size)
        return ans
        
        ans = results.reshape(batch_size, 16, 16, 16, 16).transpose(2, 3).reshape(batch_size, self.image_size, self.image_size)
        return ans
        
        ans = torch.FloatTensor(batch_size, self.image_size, self.image_size).zero_()
        
        n = int(self.image_size/self.patch_size)
        for k in range(self.patch_num):
            a1 = k%n
            b1 = int(k/n)
            ans[:, b1*self.patch_size:(b1+1)*self.patch_size, a1*self.patch_size:(a1+1)*self.patch_size] = results[:, k, :].reshape(batch_size, self.patch_size, self.patch_size)
        return ans.to(results.device)
    
    def forward(self, inputs, rain_p, labels = None, mask = None):
        
        batch_size  = rain_p.shape[0]
        rain_vector = self.rain_vec(rain_p)
        rain_vector = self.leakyrelu(rain_vector)
        rain_vector = rain_vector.reshape((batch_size, self.num_rain_patch, self.hidden_size))
        
        bool_masked_pos          = None
        interpolate_pos_encoding = None
        head_mask                = None
        output_attentions        = None
        output_hidden_states     = None
        return_dict              = None
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        embedding_output = self.model.embeddings(
            pixel_values=inputs, rain_p=rain_vector, 
            bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )
        encoder_outputs = self.model.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.model.layernorm(sequence_output)
        outputs = sequence_output[:, 1:-self.num_rain_patch, :]
        results = self.linear(outputs)
        results = self.vec2img(results)
        
        if labels is not None:
            labels   = torch.squeeze(labels, 3)
                
            mseloss  = self.MSELoss(results, labels)
        else:
            mseloss = None
        output = {
            "loss": mseloss,
            "results": results
        }
        return output
