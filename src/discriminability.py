
''' discriminability.py : finding discriminability and masking based off of it '''

import torch 
from sklearn.metrics import roc_auc_score


def get_head_auc(reps_eval_pos, reps_eval_neg, meandiff, num_heads, head_dim, groups_heads, kv_heads, layer, head, op = "attn_output"):

    ''' gives AUC for an attention head '''

    count_heads = num_heads if op in ('query', 'attn_output') else kv_heads
    bsz_pos_val, bsz_neg_val = len(reps_eval_pos[op][layer]), len(reps_eval_neg[op][layer])

    # Reshape Data
    val_features_pos = reps_eval_pos[op][layer].view(bsz_pos_val, 1, count_heads, head_dim).transpose(1, 2).squeeze()[:,...]
    val_features_neg = reps_eval_neg[op][layer].view(bsz_neg_val, 1, count_heads, head_dim).transpose(1, 2).squeeze()[:,...]
    
    # Reshape Vectors
    vectors_layer    = meandiff[op][layer] # [d]
    vectors_layer    = vectors_layer.view(1, 1, count_heads, head_dim).transpose(1,2).squeeze()

    X_pos_val = val_features_pos[:,head,:] 
    X_neg_val = val_features_neg[:,head,:]
    X_val     = torch.concat([X_neg_val, X_pos_val])
    Y_val     = [0] * len(X_neg_val) + [1] * len(X_pos_val)

    head_vector = vectors_layer[head,:] # head specific vector 

    Scores_val = (X_val @ head_vector).float().numpy()

    scores_neg = Scores_val[:len(X_neg_val)]
    scores_pos = Scores_val[len(X_neg_val):]

    auc = roc_auc_score(Y_val, Scores_val)
            
    return auc, scores_pos, scores_neg


def mask_heads_meandiff(lhKeepList, meandiff, count_heads, head_dim, op = "attn_output"):

    ''' masks all bbut the most discriminative heads based on a list of layer/head combinations to keep '''

    layer_to_head_keep = { L : [] for L,_ in  lhKeepList }

    for L, H in lhKeepList:
        layer_to_head_keep[L].append(H)

    for layer in layer_to_head_keep:
        
        h_keep            = layer_to_head_keep[layer]
        meandiff_reshaped = meandiff[op][layer].view(1, count_heads, head_dim)
        mask              = torch.zeros_like(meandiff_reshaped)
        mask[:,h_keep,:]  = 1
        meandiff_reshaped *= mask
        meandiff_reshaped = meandiff_reshaped.view(1,-1)

        meandiff[op][layer] = meandiff_reshaped
    
    return meandiff