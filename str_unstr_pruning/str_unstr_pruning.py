from ultralytics import YOLO
from ultralytics import settings
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck
from ultralytics.utils.torch_utils import initialize_weights
import os
import torch.nn.utils.prune as prune
import random
import torch
import torch.nn as nn
import torch_pruning as tp
from torchinfo import summary
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WANDB_DISABLED'] = 'true'


def infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add

class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        # y = list(self.cv1(x).chunk(2, 1))
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

def transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    state_dict = c2f.state_dict()
    state_dict_v2 = c2f_v2.state_dict()

    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]

    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]

    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    c2f_v2.load_state_dict(state_dict_v2)

def replace_c2f_with_c2f_v2(module):
    for name, child_module in module.named_children():
        if isinstance(child_module, C2f):
            # Replace C2f with C2f_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels, child_module.cv2.conv.out_channels,
                            n=len(child_module.m), shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)
            transfer_weights(child_module, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f_with_c2f_v2(child_module)

ROOT_DIR = '/path/to/robotflow'


model = YOLO('/path/to/runs/detect/project_name/weights/best.pt')
replace_c2f_with_c2f_v2(model.model)
initialize_weights(model.model)
#summary(model.model, input_size=(1, 3, 640, 640))
macs_list=[]
base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, torch.randn(1, 3, 640, 640))
macs_list.append(base_macs)
parameters_to_prune = []


for module in model.model.modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        parameters_to_prune.append((module, 'weight'))
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.4, 
)

for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

##structure pruning
def count_zero_weights_per_channel(weights):
    # Reshape to (out_channels, -1) to simplify channel-wise operations
    reshaped_weights = weights.view(weights.size(0), -1)

    # Count the number of zero elements per channel
    zero_count_per_channel = (reshaped_weights == 0).sum(dim=1).float()

    # Calculate the percentage of zeros per channel
    total_elements_per_channel = reshaped_weights.size(1)
    zero_percentage = (zero_count_per_channel / total_elements_per_channel) * 100

    return zero_percentage


def get_module_by_name(model, name):
    """
    Traverse a model using a dot-separated layer name, converting numeric parts to indices.

    Args:
        model (torch.nn.Module): The base model.
        name (str): The string path to the desired submodule (e.g., 'model.2.cv1.bn').

    Returns:
        torch.nn.Module: The nested submodule.
    """
    current_module = model  # Start with the base model
    parts = name.split('.')  # Split the path by dots

    for part in parts:
        # Check if the part is a number, meaning it's an index
        if part.isdigit():
            # Convert the string to an integer and access it as an index
            current_module = current_module[int(part)]
        else:
            # Otherwise, it's a submodule name; access it from _modules
            current_module = current_module._modules[part]

    return current_module

zero_percentage_by_layer = {}

for name, layer in model.model.named_modules():
    if isinstance(layer, nn.Conv2d):  # For convolutional layers
        weights = layer.weight.data
        zero_percentage = count_zero_weights_per_channel(weights)

        # Store zero percentage with layer name
        zero_percentage_by_layer[name] = zero_percentage
zero_weight_percentage=[]
for layer_name, zero_percentage in zero_percentage_by_layer.items():
    # Sort indices by zero percentage in descending order
    zero_weight_percentage.append(zero_percentage)
zero_weight_percentage=torch.cat(zero_weight_percentage)
sort_zero_weight_percentage,_=torch.sort(zero_weight_percentage,descending=True)
threshold=sort_zero_weight_percentage[int(len(sort_zero_weight_percentage)*0.2)]




for name, param in model.model.named_parameters():
    param.requires_grad = True


DG = tp.DependencyGraph()

DG.build_dependency(model.model, example_inputs=torch.randn( 1,3, 640, 640))
index_dict={}
for layer_name, zero_percentage in zero_percentage_by_layer.items():
    nested_module = get_module_by_name(model.model, layer_name)

    if sum(zero_percentage>=threshold)>0: 

        if sum(zero_percentage >= threshold) == zero_percentage.shape[0]:
            index_ = random.sample((zero_percentage >= threshold).nonzero(as_tuple=True)[0].tolist(),
                                   int(zero_percentage.shape[0] * 0.95))
            index_=sorted(index_)
        else:
            index_ = (zero_percentage >= threshold).nonzero(as_tuple=True)[0].tolist()
        if '22' not in layer_name: # and '21.m.0.cv2.conv' not in layer_name:

            pruning_group = DG.get_pruning_group(nested_module, tp.prune_conv_out_channels,idxs=index_)
            pruning_group.prune()

initialize_weights(model.model)
pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model.model, torch.randn(1, 3, 640, 640))
current_speed_up = float(macs_list[0]) / pruned_macs
print(f"After pruning iter: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M," 
              f" speed up={current_speed_up}")

model.ckpt.update(dict(model=model.model))
del model.ckpt["ema"]
torch.save(model.model,"pruned_unstr_str_40_20.pt")
model.model=torch.load('pruned_unstr_str_40_20.pt')
results = model.train(data="/path/to/robotflow/data.yaml", epochs=200,batch=32,device=4,project='finetune',name='str_unstr_pruning2040_remove')


metrics = model.val()


model.export(format='onnx')
