import transformer_lens
from transformer_lens import HookedTransformer, utils
import torch
import numpy as np
import gradio as gr
import pprint
import json
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from IPython.display import HTML
from functools import partial
import tqdm.notebook as tqdm
import plotly.express as px
import pandas as pd

import gradio as gr

from html import escape
import colorsys

from IPython.display import display


cfg = {
        "seed": 49,
        "batch_size": 4096,
        "buffer_mult": 384,
        "lr": 1e-4,
        "num_tokens": int(2e9),
        "l1_coeff": 3e-4,
        "beta1": 0.9,
        "beta2": 0.99,
        "dict_mult": 8,
        "seq_len": 128,
        "d_mlp": 2048,
        "enc_dtype":"fp32",
        "remove_rare_dir": False,
        }

cfg["model_batch_size"] = 64
cfg["buffer_size"] = cfg["batch_size"] * cfg["buffer_mult"]
cfg["buffer_batches"] = cfg["buffer_size"] // cfg["seq_len"]


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = cfg["d_mlp"] * cfg["dict_mult"]
        d_mlp = cfg["d_mlp"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_mlp, d_hidden, dtype=dtype)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_mlp, dtype=dtype)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_mlp, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

        self.to("cuda")

    def forward(self, x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

    # def get_version(self):
    #     return 1+max([int(file.name.split(".")[0]) for file in list(SAVE_DIR.iterdir()) if "pt" in str(file)])

    # def save(self):
    #     version = self.get_version()
    #     torch.save(self.state_dict(), SAVE_DIR/(str(version)+".pt"))
    #     with open(SAVE_DIR/(str(version)+"_cfg.json"), "w") as f:
    #         json.dump(cfg, f)
    #     print("Saved as version", version)

    # def load(cls, version):
    #     cfg = (json.load(open(SAVE_DIR/(str(version)+"_cfg.json"), "r")))
    #     pprint.pprint(cfg)
    #     self = cls(cfg=cfg)
    #     self.load_state_dict(torch.load(SAVE_DIR/(str(version)+".pt")))
    #     return self

    @classmethod
    def load_from_hf(cls, version):
        """
        Loads the saved autoencoder from HuggingFace.

        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """
        if version=="run1":
            version = 25
        elif version=="run2":
            version = 47

        cfg = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json")
        pprint.pprint(cfg)
        self = cls(cfg=cfg)
        self.load_state_dict(utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True))
        return self

def replacement_hook(mlp_post, hook, encoder):
    mlp_post_reconstr = encoder(mlp_post)[1]
    return mlp_post_reconstr

def mean_ablate_hook(mlp_post, hook):
    mlp_post[:] = mlp_post.mean([0, 1])
    return mlp_post

def zero_ablate_hook(mlp_post, hook):
    mlp_post[:] = 0.
    return mlp_post

@torch.no_grad()
def get_recons_loss(num_batches=5, local_encoder=None):
    if local_encoder is None:
        local_encoder = encoder
    loss_list = []
    for i in range(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
        loss = model(tokens, return_type="loss")
        recons_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), partial(replacement_hook, encoder=local_encoder))])
        # mean_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), mean_ablate_hook)])
        zero_abl_loss = model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(utils.get_act_name("post", 0), zero_ablate_hook)])
        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(f"loss: {loss:.4f}, recons_loss: {recons_loss:.4f}, zero_abl_loss: {zero_abl_loss:.4f}")
    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    print(f"Reconstruction Score: {score:.2%}")
    # print(f"{((zero_abl_loss - mean_abl_loss)/(zero_abl_loss - loss)).item():.2%}")
    return score, loss, recons_loss, zero_abl_loss

# Frequency
@torch.no_grad()
def get_freqs(num_batches=25, local_encoder=None):
    if local_encoder is None:
        local_encoder = encoder
    act_freq_scores = torch.zeros(local_encoder.d_hidden, dtype=torch.float32).cuda()
    total = 0
    for i in tqdm.trange(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]

        _, cache = model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
        mlp_acts = cache[utils.get_act_name("post", 0)]
        mlp_acts = mlp_acts.reshape(-1, d_mlp)

        hidden = local_encoder(mlp_acts)[2]

        act_freq_scores += (hidden > 0).sum(0)
        total+=hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores==0).float().mean()
    print("Num dead", num_dead)
    return act_freq_scores



SPACE = "·"
NEWLINE="↩"
TAB = "→"

def create_html(strings, values, max_value=None, saturation=0.5, allow_different_length=False, return_string=False):
    # escape strings to deal with tabs, newlines, etc.
    escaped_strings = [escape(s, quote=True) for s in strings]
    processed_strings = [
        s.replace("\n", f"{NEWLINE}<br/>").replace("\t", f"{TAB}&emsp;").replace(" ", "&nbsp;")
        for s in escaped_strings
    ]

    if isinstance(values, torch.Tensor) and len(values.shape)>1:
        values = values.flatten().tolist()

    if not allow_different_length:
        assert len(processed_strings) == len(values)

    # scale values
    if max_value is None:
        max_value = max(max(values), -min(values))+1e-3
    scaled_values = [v / max_value * saturation for v in values]

    # create html
    html = ""
    for i, s in enumerate(processed_strings):
        if i<len(scaled_values):
            v = scaled_values[i]
        else:
            v = 0
        if v < 0:
            hue = 0  # hue for red in HSV
        else:
            hue = 0.66  # hue for blue in HSV
        rgb_color = colorsys.hsv_to_rgb(
            hue, v, 1
        )  # hsv color with hue 0.66 (blue), saturation as v, value 1
        hex_color = "#%02x%02x%02x" % (
            int(rgb_color[0] * 255),
            int(rgb_color[1] * 255),
            int(rgb_color[2] * 255),
        )
        html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{s}</span>'
    if return_string:
        return html
    else:
        display(HTML(html))

def basic_feature_vis(text, feature_index, max_val=0):
    feature_in = encoder.W_enc[:, feature_index]
    feature_bias = encoder.b_enc[feature_index]
    _, cache = model.run_with_cache(text, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
    mlp_acts = cache[utils.get_act_name("post", 0)][0]
    feature_acts = F.relu((mlp_acts - encoder.b_dec) @ feature_in + feature_bias)
    if max_val==0:
        max_val = max(1e-7, feature_acts.max().item())
        # print(max_val)
    # if min_val==0:
    #     min_val = min(-1e-7, feature_acts.min().item())
    return basic_token_vis_make_str(text, feature_acts, max_val)
def basic_token_vis_make_str(strings, values, max_val=None):
    if not isinstance(strings, list):
        strings = model.to_str_tokens(strings)
    values = utils.to_numpy(values)
    if max_val is None:
        max_val = values.max()
    # if min_val is None:
    #     min_val = values.min()
    header_string = f"<h4>Max Range <b>{values.max():.4f}</b> Min Range: <b>{values.min():.4f}</b></h4>"
    header_string += f"<h4>Set Max Range <b>{max_val:.4f}</b></h4>"
    # values[values>0] = values[values>0]/ma|x_val
    # values[values<0] = values[values<0]/abs(min_val)
    body_string = create_html(strings, values, max_value=max_val, return_string=True)
    return header_string + body_string
# display(HTML(basic_token_vis_make_str(tokens[0, :10], mlp_acts[0, :10, 7], 0.1)))
# # %%
# The `with gr.Blocks() as demo:` syntax just creates a variable called demo containing all these components



try:
    demos[0].close()
except:
    pass
demos = [None]
def make_feature_vis_gradio(feature_id, starting_text=None, batch=None, pos=None):
    if starting_text is None:
        starting_text = model.to_string(all_tokens[batch, 1:pos+1])
    try:
        demos[0].close()
    except:
        pass
    with gr.Blocks() as demo:
        gr.HTML(value=f"Hacky Interactive Neuroscope for gelu-1l")
        # The input elements
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text", value=starting_text)
                # Precision=0 makes it an int, otherwise it's a float
                # Value sets the initial default value
                feature_index = gr.Number(
                    label="Feature Index", value=feature_id, precision=0
                )
                # # If empty, these two map to None
                max_val = gr.Number(label="Max Value", value=None)
                # min_val = gr.Number(label="Min Value", value=None)
                inputs = [text, feature_index, max_val]
        with gr.Row():
            with gr.Column():
                # The output element
                out = gr.HTML(label="Neuron Acts", value=basic_feature_vis(starting_text, feature_id))
        for inp in inputs:
            inp.change(basic_feature_vis, inputs, out)
    demo.launch(share=True)
    demos[0] = demo


SPACE = "·"
NEWLINE="↩"
TAB = "→"
def process_token(s):
    if isinstance(s, torch.Tensor):
        s = s.item()
    if isinstance(s, np.int64):
        s = s.item()
    if isinstance(s, int):
        s = model.to_string(s)
    s = s.replace(" ", SPACE)
    s = s.replace("\n", NEWLINE+"\n")
    s = s.replace("\t", TAB)
    return s

def process_tokens(l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [process_token(s) for s in l]

def process_tokens_index(l):
    if isinstance(l, str):
        l = model.to_str_tokens(l)
    elif isinstance(l, torch.Tensor) and len(l.shape)>1:
        l = l.squeeze(0)
    return [f"{process_token(s)}/{i}" for i,s in enumerate(l)]

def create_vocab_df(logit_vec, make_probs=False, full_vocab=None):
    if full_vocab is None:
        full_vocab = process_tokens(model.to_str_tokens(torch.arange(model.cfg.d_vocab)))
    vocab_df = pd.DataFrame({"token": full_vocab, "logit": utils.to_numpy(logit_vec)})
    if make_probs:
        vocab_df["log_prob"] = utils.to_numpy(logit_vec.log_softmax(dim=-1))
        vocab_df["prob"] = utils.to_numpy(logit_vec.softmax(dim=-1))
    return vocab_df.sort_values("logit", ascending=False)

def list_flatten(nested_list):
    return [x for y in nested_list for x in y]
def make_token_df(tokens, len_prefix=5, len_suffix=1):
    str_tokens = [process_tokens(model.to_str_tokens(t)) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] for str_tok in str_tokens]

    context = []
    batch = []
    pos = []
    label = []
    for b in range(tokens.shape[0]):
        # context.append([])
        # batch.append([])
        # pos.append([])
        # label.append([])
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p-len_prefix):p])
            if p==tokens.shape[1]-1:
                suffix = ""
            else:
                suffix = "".join(str_tokens[b][p+1:min(tokens.shape[1]-1, p+1+len_suffix)])
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
            batch.append(b)
            pos.append(p)
            label.append(f"{b}/{p}")
    # print(len(batch), len(pos), len(context), len(label))
    return pd.DataFrame(dict(
        str_tokens=list_flatten(str_tokens),
        unique_token=list_flatten(unique_token),
        context=context,
        batch=batch,
        pos=pos,
        label=label,
    ))

def argsort(seq, reverse=False):
        return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)

def get_logit_effect(model, autoencoder, feature_id):
    return autoencoder.W_dec[feature_id] @ model.W_out[0] @ model.W_U

def get_hidden_act(hidden_activations, sorted_feature_idxs, activation_rank):
    idx = sorted_feature_idxs[activation_rank]
    return utils.to_numpy(hidden_acts[:, idx])

def _feature_search_worker(dataset, model, autoencoder, batch_size=128, token_length=128):
    '''
    Search through features of a pre-trained SAE.
    'dataset' must first be loaded as an HuggingFace Dataset:
    from 
    dataset = load_dataset('file_type', data_files="local_file.file_type", split="whatever_split")
    '''
    tokenized_data = utils.tokenize_and_concatenate(dataset, model.tokenizer, max_length=token_length)
    tokenized_data = tokenized_data.shuffle(seed=42)
    all_tokens = tokenized_data["tokens"]

    tokens = all_tokens[:batch_size] 

    _, cache = model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
    mlp_acts = cache[utils.get_act_name("post", 0)]
    mlp_acts_flattened = mlp_acts.reshape(-1, cfg["d_mlp"])

    loss, x_reconstruct, hidden_acts, l2_loss, l1_loss = autoencoder(mlp_acts_flattened)
    batch_df = make_token_df(tokens)

    # Go through every feature and keep track of the most activated feature
    most_activations = [None for _ in range(autoencoder.d_hidden)]
    for feature_id in range(autoencoder.d_hidden):

        feature_act = utils.to_numpy(hidden_acts[:, feature_id])

        # Measure most activation based on proportion of the top 3 activations
        sorted_feature_idxs = np.argsort(feature_act)
        sorted_feature_act = feature_act[sorted_feature_idxs]
        act_sum = sorted_feature_act.sum()
        if act_sum == 0:
            most_activations[feature_id] = -np.inf
        else:
            most_activations[feature_id] = sorted_feature_act[-3:].sum() / sorted_feature_act.sum()

        # Measure most activation based on average activation (for now)
        # most_activations[feature_id] = feature_act.mean()

    # Sort the features by the mean of the max activations
    sorted_feature_idxs = argsort(most_activations, reverse=True)

    return batch_df, sorted_feature_idxs, hidden_acts

def feature_search(path, model, encoder, batch_size=128, token_length=128):
    
    dataset = load_dataset('csv', data_files=path, split="train")
    dataset = dataset.rename_column('content', 'text')
    dataset = dataset.filter(lambda x: x['text'] != None)

    # Do custom filtering here
    segment = dataset.filter(lambda x: x['sentiment'] == 'surprise')

    batch_df, sorted_idxs, hidden_acts = _feature_search_worker(segment, model, encoder)

    return batch_df, sorted_idxs, hidden_acts


def keyword_search(model, autoencoder, keyword):
    '''
    Search through features of a pre-trained SAE.
    'dataset' must first be loaded as an HuggingFace Dataset:
    from 
    dataset = load_dataset('file_type', data_files="local_file.file_type", split="whatever_split")
    '''

    #logit_effects = [autoencoder.W_dec[feature_id] @ model.W_out[0] @ model.W_U
    #                 for feature_id in range(autoencoder.d_hidden)]
    logit_effects = []
    for feature_id in range(autoencoder.d_hidden):
        logit_effects.append(autoencoder.W_dec[feature_id] @ model.W_out[0] @ model.W_U)
    
    vocab_dfs = [create_vocab_df(logit_effect) for logit_effect in logit_effects]

    with_keyword = []
    for vocab in vocab_dfs:
        # Include feature if the keyword is in the top 10
        if keyword in vocab['token'].head(10).values:
            with_keyword.append(vocab)

    return with_keyword

if __name__ == '__main__':

    #global cfg 

    model = HookedTransformer.from_pretrained("gelu-1l").to(DTYPES[cfg["enc_dtype"]])
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    d_mlp = model.cfg.d_mlp
    d_vocab = model.cfg.d_vocab

    data = load_dataset("NeelNanda/c4-code-20k", split="train")
    tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
    tokenized_data = tokenized_data.shuffle(42)
    all_tokens = tokenized_data["tokens"]

    auto_encoder_run = "run1" # @param ["run1", "run2"]
    encoder = AutoEncoder.load_from_hf(auto_encoder_run)

    _ = get_recons_loss(num_batches=5, local_encoder=encoder)

    freqs = get_freqs(num_batches = 50, local_encoder = encoder)

    # Add 1e-6.5 so that dead features show up as log_freq -6.5
    log_freq = (freqs + 10**-6.5).log10()
    #px.histogram(utils.to_numpy(log_freq), title="Log Frequency of Features", histnorm='percent')

    is_rare = freqs < 1e-4
    rare_enc = encoder.W_enc[:, is_rare]
    rare_mean = rare_enc.mean(-1)
    #px.histogram(utils.to_numpy(rare_mean @ encoder.W_enc / rare_mean.norm() / encoder.W_enc.norm(dim=0)), title="Cosine Sim with Ave Rare Feature", color=utils.to_numpy(is_rare), 
    #            labels={"color": "is_rare", "count": "percent", "value": "cosine_sim"}, marginal="box", histnorm="percent", barmode='overlay')

    # Load the dataset as an HG dataset
    path = r'/home/jovyan/1L-Sparse-Autoencoder/tweet_emotions.csv'

    batch_df, sorted_idxs, hidden_acts = feature_search(path, model, encoder)

    batch_df['feature'] = get_hidden_act(hidden_acts, sorted_idxs, 0)
    batch_df_love.sort_values("feature", ascending=False).head(20).style.background_gradient("coolwarm")

    #with_keyword = keyword_search(model, encoder, 'happy')
    


    

