''' steering.py : create and steering with, steering vectors'''

import torch 
import tqdm
from functools import reduce
import copy 
from utils import get_logits
from discriminability import mask_heads_meandiff


def t_(x, **kwargs):

    ''' translates representation '''

    alpha        = kwargs.get('alpha')
    meandiff     = kwargs.get('meandiff')
    
    x.add_(alpha * meandiff)

    return x

class PreHookExtract:

    ''' extract representation input''' 

    def __init__(self, agg = 'last'):
        self.saved_values        = None  
        self.agg                 = agg

    def __call__(self, module, input):
        if self.agg == 'last':
            self.saved_values = input[0][:,-1,:].cpu() 
        elif isinstance(self.agg, int):
            self.saved_values = input[0][:,self.agg,:].cpu() 
        elif self.agg == None:
            self.saved_values = input[0][:,:,:].cpu() 
        return input  
        
class ForwardHookExtract:

    ''' extract representation output ''' 

    def __init__(self, agg = 'last'):
        self.saved_values        = None  
        self.agg                 = agg 

    def __call__(self, module, input, output):

        if isinstance(output, tuple): 
            self.saved_values = output[0][:,:,:]
        else:
            self.saved_values = output[:,:,:]

        if self.agg == 'last':
            self.saved_values = self.saved_values[:,-1,:].cpu()
        elif isinstance(self.agg, int):
            self.saved_values = self.saved_values[:,self.agg,:].cpu()
        elif self.agg == None:
            self.saved_values = self.saved_values[:,:,:].cpu()

        return output  

###########################################################################
##                       Steering Vector Creation                        ##
###########################################################################

def hook_reps_extract(model, operations, op_dict, layers = list(range(0,10)), agg = 'last'):

    ''' hook representations for saving ''' 

    hooks, handles = {}, {} 

    for layer_id in layers:

        hooks[layer_id], handles[layer_id] = {}, {}

        for op in operations:

            assert op_dict[op]['hook type'] in ['forward', 'forward_pre']; "Error: hook type must be forward or forward_pre"

            op_exact_name = op_dict[op]['module'].format(layer = str(layer_id))
            module        = reduce(getattr, op_exact_name.split("."), model)

            if op_dict[op]['hook type'] == 'forward':
                hook                  = ForwardHookExtract(agg = agg)
                handle                = module.register_forward_hook(hook)

            elif op_dict[op]['hook type'] == 'forward_pre':
                hook                  = PreHookExtract(agg = agg)
                handle                = module.register_forward_pre_hook(hook)

            hooks[layer_id][op]   = hook
            handles[layer_id][op] = handle

    return hooks, handles 

def extract_reps(model, operations, layers, dataloader, tokenizer, op_dict, add_special_tokens = False, agg = 'last'):

    ''' last token representation extraction '''

    hooks, handles = hook_reps_extract(model, operations, op_dict, layers, agg = agg)
    reps = {op : {layer : [] for layer in layers} for op in operations}

    for batch in tqdm.tqdm(dataloader, desc = "Representation Extraction"):
        inputs = tokenizer(batch, return_tensors = 'pt', padding = True, add_special_tokens = add_special_tokens) 

        with torch.no_grad():

            model(inputs['input_ids'].cuda(), attention_mask=inputs["attention_mask"].cuda()) # populate 

            for layer in layers:  
                for op in operations:
                    reps[op][layer].append(hooks[layer][op].saved_values) # save

    unhook(handles)

    for op in reps:     # for each op + layer combo, list of tensors --> tensor
        for layer in layers:
            reps[op][layer] = torch.vstack(reps[op][layer])

    return reps

def mean_difference(operations, pos_representations, neg_representations):

    ''' creates MD steering vectors for all layers ''' 

    op_to_layer_to_meandiff = {op : {} for op in operations}
    for op in operations:
        temp_pos, temp_neg = pos_representations[op], neg_representations[op]

        for layer in temp_pos.keys():
            pos_repr, neg_repr = temp_pos[layer], temp_neg[layer]
            meandiff = (pos_repr.mean(dim = 0, keepdim = True) - neg_repr.mean(dim = 0, keepdim = True))
            op_to_layer_to_meandiff[op][layer] = meandiff

    return op_to_layer_to_meandiff

###########################################################################
##                       Steering Vector Injection                       ##
###########################################################################

class PreHookSteer:

    ''' injects steering vector into input of a function ''' 

    def __init__(self, inject_op = None, inject_params = {}):

        self.inject_op           = inject_op 
        self.inject_params       = inject_params  

    def __call__(self, module, input):

        self.inject_op(input[0], **self.inject_params) 
        return input  

class ForwardHookSteer:

    ''' injects steering vector into output of a function ''' 

    def __init__(self, inject_op = None, inject_params = {}):

        self.inject_op           = inject_op 
        self.inject_params       = inject_params  

    def __call__(self, module, input, output):

        if isinstance(output, tuple):
            self.inject_op(output[0], **self.inject_params)  
        else:
            self.inject_op(output, **self.inject_params)

        return output 


def hook_model_steer(model, operations, op_layer_params, op_dicts):

    ''' registers injection hooks for desired operations and positions '''

    hooks, handles   = {}, {}

    for op in operations:

        hooks[op], handles[op] = {},{}

        for layer in op_layer_params[op]: 

            op_exact_name = op_dicts[op]['module'].format(layer = str(layer))
            module        = reduce(getattr, op_exact_name.split("."), model)
            inject_op     = op_layer_params[op][layer]['inject_op'] # i.e. "translation_op_"
            inject_params = op_layer_params[op][layer]['inject_params']

            if op_dicts[op]['hook type'] == "forward_pre":
                hook          = PreHookSteer(inject_op = inject_op, inject_params = inject_params)
                handle        = module.register_forward_pre_hook(hook)

            else:
                hook          = ForwardHookSteer(inject_op = inject_op, inject_params = inject_params)
                handle        = module.register_forward_hook(hook)
            
            hooks[op][layer], handles[op][layer] = hook, handle

    return hooks, handles

def logit_grading_steered(model, tokenizer, op_dict, data, split, add_special_tokens, position = -1, operations = ["attn_output"], layer = 10, head = 10, alpha = 30, meandiff = None, num_heads = 32, head_dim = 128):

    # masks MD Vector so only the specific layer-head combination is steered with the appropriate SV
    meandiff_masked     = mask_heads_meandiff([(layer, head)], copy.deepcopy(meandiff), num_heads, head_dim) 

    op_layer_params = {"attn_output" : { layer: 
                            {'inject_op' : t_, 
                            'inject_params' : {'alpha'   : alpha, 
                                                'meandiff' : meandiff_masked["attn_output"][layer].squeeze().to(model.device)}
                            } 
                        } }

    hooks, handles = hook_model_steer(model, operations, op_layer_params, op_dict)
    logits         = get_logits(data[split]['gen_dataloader'], model, tokenizer, position = position, add_special_tokens = add_special_tokens)
    unhook(handles)

    return logits 
    
def unhook(handles):

    ''' removes hooks '''

    for op in handles.keys(): # unhook
        for layer in handles[op].keys():
            handles[op][layer].remove()