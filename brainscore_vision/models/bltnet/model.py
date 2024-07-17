
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from brainscore_vision.model_helpers.check_submission import check_models
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_kietzmannlab.torch_models import LayerNorm
import os
import json

LAYERS = ['LNRCL.RCL_layer_norm9']


                

def get_model(name):
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    preprocessing =  functools.partial(load_preprocess_images, image_size=128, normalize_mean=(0.4987, 0.4702, 0.4050), normalize_std=(0.2711, 0.2635, 0.2810))
    
    model_path = '/share/klab/vkapoor/ms_coco_embeddings_pth'
    
    model_name = 'blt_ms_coco_embeddings'
    
    weight_file_path = os.path.join(model_path, model_name + '.pth')
    
    model_json = os.path.join(model_path, model_name + '.json')
    
    with open(model_json) as file:
                    checkpoint_data = json.load(file)
    checkpoint_n_classes = checkpoint_data["num_classes"]
    checkpoint_n_recurrent_steps = checkpoint_data["n_recurrent_steps"]
    divide_n_channels = checkpoint_data["divide_n_channels"]
    double_decker = checkpoint_data["double_decker"]
    
    blt_model = BLT_net_macro(checkpoint_n_classes, 
                              checkpoint_n_recurrent_steps, 
                              double_decker=double_decker,
                              divide_n_channels=divide_n_channels)
    
    blt_model.load_state_dict(torch.load(weight_file_path, map_location=device))
    
    wrapper = PytorchWrapper(identifier=name, model=blt_model, preprocessing=preprocessing)

    
    return wrapper

class BLT_net_macro(nn.Module):
    def __init__(
        self,
        num_classes,
        num_recurrent_steps,
        double_decker=False,
        divide_n_channels=1,
        norm_type="LN",
        l_flag=True,
        t_flag=True,
        lt_interact=0,
        LT_position="all",
        classifier_bias=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_recurrent_steps = num_recurrent_steps
        self.double_decker = double_decker
        self.divide_n_channels = divide_n_channels
        self.norm_type = norm_type
        self.l_flag = l_flag
        self.t_flag = t_flag
        self.lt_interact = lt_interact
        self.n_timesteps = max(self.num_recurrent_steps, 1)
        self.use_bias = True if norm_type == "no_norm" else False
        lt_flag_prelast = 1
        if LT_position == "last":
            lt_flag_prelast = 0

        if not self.double_decker:
            self.layer_sizes = [3, 32, 64, 128, 256, 512]
            self.kernel_sizes = [7, 5, 3, 3, 3]
            self.pooling_layers = [2, 3, 4]
        else:
            self.layer_sizes = [
                3,
                64,
                64,
                128,
                128,
                256,
                256,
                512,
                512,
                1024,
                1024,
            ]
            self.kernel_sizes = [7, 7, 5, 5, 3, 3, 3, 3, 1, 1]
            self.pooling_layers = [2, 3, 8, 9]

        self.RCL = nn.ModuleDict()
        self.LNRCL = nn.ModuleDict()

        for i in range(len(self.kernel_sizes)):
            if i in self.pooling_layers:
                pool_input = True
            else:
                pool_input = False
            if i == len(self.kernel_sizes) - 1:
                self.t_flag = False
            self.RCL[f"RCL_{i}"] = BLT_ConvLayer(
                self.layer_sizes[i] // divide_n_channels
                if i > 0
                else self.layer_sizes[i],
                self.layer_sizes[i + 1] // divide_n_channels,
                2 * self.layer_sizes[i + 1] // divide_n_channels
                if not double_decker
                else self.layer_sizes[i + 1] // divide_n_channels
                if i % 2 == 0
                else 2 * self.layer_sizes[i + 1] // divide_n_channels,
                self.kernel_sizes[i],
                use_bias=self.use_bias,
                pool_input=pool_input,
                l_flag=self.l_flag * lt_flag_prelast,
                t_flag=self.t_flag * lt_flag_prelast,
                lt_interact=self.lt_interact,
            )
            self.LNRCL[f"RCL_layer_norm{i}"] = LayerNorm(
                self.layer_sizes[i + 1] // divide_n_channels
            )


        self.readout_dense = nn.Linear(
            self.layer_sizes[-1] // divide_n_channels,
            num_classes,
            bias=classifier_bias,
        )

    def forward(self, inputs):
        n_layers = len(self.RCL)
        activations = [
            [None] * (n_layers + 1) for _ in range(self.n_timesteps)
        ]
        outputs = [None] * self.n_timesteps

        for t in range(self.n_timesteps):
            n = 0
            layer = self.RCL[f"RCL_{n}"]
            layer_norm = self.LNRCL[f"RCL_layer_norm{n}"]
            activations[t][n] = F.relu(layer_norm(layer(inputs)))

            for n in range(1, len(self.kernel_sizes)):
                layer = self.RCL[f"RCL_{n}"]
                layer_norm = self.LNRCL[f"RCL_layer_norm{n}"]
                if t == 0:
                    b_input = activations[t][n - 1]
                    activations[t][n] = F.relu(layer_norm(layer(b_input)))
                if t > 0:
                    if n < n_layers - 1:
                        b_input = activations[t][n - 1]
                        l_input = activations[t - 1][n]
                        t_input = activations[t - 2][n + 1] if t >= 2 else None

                        activations[t][n] = F.relu(
                            layer_norm(layer(b_input, l_input, t_input))
                        )
                    else:
                        b_input = activations[t][n - 1]
                        l_input = activations[t - 1][n]
                        activations[t][n] = F.relu(
                            layer_norm(layer(b_input, l_input))
                        )
            n = len(self.kernel_sizes)
            x = F.adaptive_avg_pool2d(activations[t][len(self.kernel_sizes)-1], (1, 1))
            x = x.view(x.size(0), -1)
            outputs[t] = self.readout_dense(x)

        return outputs, activations


class BLT_ConvLayer(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        top_down_channels,
        kernel_size,
        use_bias,
        stride=1,
        pool_input=True,
        l_flag=True,
        t_flag=True,
        lt_interact=0,
    ):
        super().__init__()
        self.l_flag = l_flag
        self.t_flag = t_flag
        self.lt_interact = lt_interact
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.top_down_channels = top_down_channels
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.stride = stride
        self.pool_input = pool_input
        self.bottom_up = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            bias=self.use_bias,
            padding="same",
        )
        if self.l_flag:
            self.lateral = nn.Conv2d(
                self.out_channels,
                self.out_channels,
                kernel_size,
                bias=self.use_bias,
                padding="same",
            )
        if self.t_flag:
            ct_padding = int((self.kernel_size - 1) / 2)
            self.top_down = nn.ConvTranspose2d(
                self.top_down_channels,
                self.out_channels,
                self.kernel_size,
                stride=2,
                bias=self.use_bias,
                padding=ct_padding,
                output_padding=1,
            )
        self.pool_input = pool_input
        if self.pool_input:
            self.pool = nn.MaxPool2d(2, 2)

    @staticmethod
    def ResizeTensorForTConv(output_shape):
        class ResizeTensorForTConv(nn.Module):
            def __init__(self, output_shape):
                super().__init__()
                self.output_shape = output_shape

            def forward(self, x):
                new_height, new_width = self.output_shape[-2:]

                x = F.interpolate(
                    x,
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                )

                return x

        return ResizeTensorForTConv(output_shape)

    def resize_t_conv(self, x, output_shape):
        self.resize_layer = self.ResizeTensorForTConv(output_shape)
        return self.resize_layer(x)

    def forward(self, b_input, l_input=None, t_input=None):
        if self.pool_input:
            b_input = self.pool(b_input)
        b_input = self.bottom_up(b_input)
        if self.l_flag:
            if l_input is not None:
                l_input = self.lateral(l_input)
        else:
            l_input = None
        if self.t_flag:
            if t_input is not None:
                t_input = self.top_down(t_input)
                t_input = self.resize_t_conv(t_input, b_input.shape)
        else:
            t_input = None

        if l_input is not None and t_input is not None:
            if self.lt_interact == 0:
                output = b_input + l_input + t_input
            else:
                output = b_input * (1.0 + l_input + t_input)
        elif l_input is not None:
            if self.lt_interact == 0:
                output = b_input + l_input
            else:
                output = b_input * (1.0 + l_input)
        elif t_input is not None:
            if self.lt_interact == 0:
                output = b_input + t_input
            else:
                output = b_input * (1.0 + t_input)
        else:
            output = b_input

        return output                
  
def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """

    return ['bltnet']  
def get_layers(name):
    
    assert name == 'bltnet'
    brainscore_layers = LAYERS
    
    return  brainscore_layers  
    
def get_bibtex(model_identifier):
    """
    A method returning the bibtex reference of the requested model as a string.
    """

    return ''    
    
if __name__ == '__main__':
    check_models.check_base_models(__name__)    