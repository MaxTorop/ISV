''' utils.py : helper functions ''' 

import torch 
import tqdm
import matplotlib.pyplot as plt
import numpy as np 

def get_logits(dataloader, model, tokenizer, position = -1, add_special_tokens = True):
    
    """ get next token logits across all elements in the dataloader """

    bar = tqdm.tqdm(dataloader, total = len(dataloader), desc = "Logit Extraction", disable = True)
    logits = [] 
    for q in bar:
        q_tok                  = tokenizer(q, return_tensors = 'pt', padding = True, add_special_tokens = add_special_tokens).to("cuda")
        with torch.no_grad():
            logits_batch       = model(q_tok['input_ids'], attention_mask = q_tok['attention_mask'], pad_token_id=tokenizer.pad_token_id).logits.detach().cpu()
        q_tok  = q_tok.to("cpu")
        logits.append(logits_batch[:,position,:])
    return torch.vstack(logits)

def viz_ips(pos_ip, neg_ip, auc, layer, head):

    ''' visualize inner products with steering vector via histogram, to show discriminability ''' 

    pos_ip = np.asarray(pos_ip)
    neg_ip = np.asarray(neg_ip)
    
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.linewidth": 1.2,
    })

    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    all_vals = np.concatenate([pos_ip, neg_ip])
    bins = np.linspace(all_vals.min(), all_vals.max(), 40)

    ax.hist(
        pos_ip,
        bins=bins,
        alpha=0.65,
        linewidth=1.5,
        histtype="stepfilled",
        label="Concept Positive Examples",
    )

    ax.hist(
        neg_ip,
        bins=bins,
        alpha=0.65,
        linewidth=1.5,
        histtype="stepfilled",
        label="Concept Negative Examples",
    )

    ax.set_xlabel("Inner Product Score")
    ax.set_ylabel("Count")
    plt.title(
        rf"Inner Product With $\mu^{{{layer},{head}}}$ "
        rf"(AUC={auc:.3f})"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(alpha=0.15)

    leg = ax.legend(frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()


def mc_grader(logits, tokenizer, GT):

    ''' grades logits against ground truth tokens ''' 

    a_token_id = tokenizer.convert_tokens_to_ids("A")
    b_token_id = tokenizer.convert_tokens_to_ids("B")
    logits_A    = logits[:, a_token_id]
    logits_B    = logits[:, b_token_id]
    predictions = (logits_B >= logits_A).int().numpy() # 0 --> A, 1 --> B
    Accuracy    = (predictions == GT).mean()
    MaxLogitIdx = logits.argmax(dim = 1).numpy()
    return Accuracy, predictions, MaxLogitIdx

def vizSingleSteering(
    results,
    layer=None,
    head=None,
    ylabel="MC concept score"
):

    ''' visualizes steering effect via alpha vs concept score ''' 

    xy = {float(alpha): results[alpha] for alpha in results}

    xs, ys = zip(*sorted(xy.items()))
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    ax.plot(xs, ys, linewidth=2)

    ax.axvline(
        x=0,
        linestyle=':',
        color='0.4',
        alpha=0.8,
        linewidth=1.3
    )

    baseline_score = xy[0.0] if 0.0 in xy else xy[0]

    ax.axhline(
        y=baseline_score,
        linestyle=':',
        color='0.4',
        alpha=0.8,
        linewidth=1.3
    )

    stats = []

    if len(stats) > 0:
        ax.text(
            0.02,
            0.02,
            "\n".join(stats),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            linespacing=1.1,
            bbox=dict(
                facecolor="white",
                alpha=0.8,
                edgecolor="none",
                pad=0.3
            )
        )

    ax.set_title(
        rf"Steering $\mu^{{{layer},{head}}}$", 
        fontsize=12,
        fontweight='bold',
        pad=6
    )

    ax.set_ylim(0, 1)
    ax.set_yticks([0.0, 0.5, 1.0])

    ax.set_xlabel(r"Steering factor $\alpha$", fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')

    ax.grid(True, alpha=0.18, linewidth=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()