''' models.py : functions for loading models ''' 

import torch 
from unsloth import FastLanguageModel
from transformers import logging as hf_logging

op_dicts = {'unsloth/Olmo-3-7B-Instruct' : { 
                        'attn_output': {
                            'hook type' : 'forward_pre',
                            'module' : 'model.layers.{layer}.self_attn.o_proj'
                        }}}

def get_unsloth_model(model_name, max_seq_length = 1024, load_in_4bit = False, load_in_8bit = False, full_finetuning = False):

    ''' load an unsloth model''' 

    hf_logging.set_verbosity_error()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,  
        load_in_8bit = load_in_8bit,
        full_finetuning = full_finetuning,
        fast_inference = False
    )
    model.eval()
    return model, tokenizer

def get_model_info(model_name):

    ''' get all relevant model info incl. loading model ''' 

    model_name = "unsloth/" + model_name
    model, tokenizer   = get_unsloth_model(model_name = model_name)
    op_dict            = { 
                            'layer output' : {
                            'hook type' : 'forward',
                            'module'    : "model.layers.{layer}"},
                            
                            'attn_output': {
                                'hook type' : 'forward_pre',
                                'module' : 'model.layers.{layer}.self_attn.o_proj'
                            }}
    L                  = len(model.model.layers) 
    add_special_tokens = True 
    num_heads    = model.config.num_attention_heads
    head_dim     = model.config.hidden_size // num_heads # double check this
    kv_heads     = model.config.num_key_value_heads
    group_heads = num_heads // kv_heads

    return model, tokenizer, op_dict, L, add_special_tokens, tokenizer, num_heads, head_dim, kv_heads, group_heads