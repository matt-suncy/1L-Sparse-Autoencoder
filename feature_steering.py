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
from rich import print as rprint

import sys         
sys.path.append(r'/home/jovyan/SAELens/home/jovyan/SAELens') 

torch.set_grad_enabled(False)
print('Disabled autograd')

model = HookedTransformer.from_pretrained("gpt2").to(torch.float32)
print('type(model)', type(model))
n_layers = model.cfg.n_layers
d_model = model.cfg.d_model
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
d_mlp = model.cfg.d_mlp
d_vocab = model.cfg.d_vocab


prompt = "Why did the chicken cross the"
answer = "road"
utils.test_prompt(prompt, answer, model)

'''
Let's plan how we are going to do the feature steering.

1. Pass the prompt through the model until the mlp_out layer (don't output it) which we'll call model_mlp_out. 
2. Pass model_mlp_out to the SAE.
3. Get the hidden layer activations from the SAE.
4. Augment the activation value of target_feature (say multiply by 5).
5. Pass the augmented hiddden activation to get an augmented reconstruction. 
6. Pass the augmented reconstruction through the model until the output layer.

'''

# Define auto encoder class

class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        #d_hidden = cfg["d_mlp"] * cfg["dict_mult"]
        d_hidden = cfg["source_model"]["value"]["hook_dimension"] * cfg["autoencoder"]["value"]["expansion_factor"]
        d_mlp = cfg["source_model"]["value"]["hook_dimension"]
        l1_coeff = cfg["loss"]["value"]["l1_coefficient"]
        dtype = torch.float32
        torch.manual_seed(cfg["random_seed"]["value"])
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
    def encode(self, x):
        x_cent = x - self.b_dec
        return F.relu(x_cent @ self.W_enc + self.b_enc)

    @torch.no_grad()
    def decode(self, hidden_acts):
        return hidden_acts @ self.W_dec + self.b_dec

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
    def load_custom_state_dict(cls, instance, state_dict, mapping):
        new_state_dict = {}
        for key in state_dict:

            if key in mapping:

                new_key = mapping[key]
                new_value = state_dict[key]
                
                
                if hasattr(instance, new_key):

                    param = getattr(instance, new_key)
                    new_value = torch.squeeze(new_value)

                    if param.shape != new_value.shape:

                        # Check if shapes are compatible for transposing
                        if param.shape == new_value.T.shape:
                            new_value = new_value.T
                        else:
                            new_value = new_value.reshape(param.shape)
                

                new_state_dict[new_key] = new_value

            else:
                new_state_dict[key] = state_dict[key]
        
        instance.load_state_dict(new_state_dict, strict=False)

    @classmethod
    def load_from_hf(cls, version):
        """
        Loads the saved autoencoder from HuggingFace.

        Version is expected to be an int, or "run1" or "run2"

        version 25 is the final checkpoint of the first autoencoder run,
        version 47 is the final checkpoint of the second autoencoder run.
        """

        #cfg = utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}_cfg.json")
        cfg = utils.download_file_from_hf("matt-suncy/sparse_autoencoder", "cfg.json")

        instance = cls(cfg=cfg)
        #self.load_state_dict(utils.download_file_from_hf("NeelNanda/sparse_autoencoder", f"{version}.pt", force_is_torch=True))
        ae_state_dict = utils.download_file_from_hf("matt-suncy/sparse_autoencoder", f"{version}.pt", force_is_torch=True).state_dict

        #self.load_state_dict(utils.download_file_from_hf("matt-suncy/sparse_autoencoder", f"{version}.pt", force_is_torch=True).state_dict)
        # Change state_dict mappings
        key_mapping = {
            #"tied_bias": "b_enc",
            #"pre_encoder_bias._bias_reference": "b_enc",
            "encoder.weight": "W_enc",
            "encoder.bias": "b_enc",
            "decoder.weight": "W_dec",
            "post_decoder_bias._bias_reference": "b_dec",
        }

        cls.load_custom_state_dict(instance, ae_state_dict, key_mapping)

        return instance
    
auto_encoder_run = 'tough-sweep-1_final' # @param ["run1", "run2"]
encoder = AutoEncoder.load_from_hf(auto_encoder_run)

def steered_forward(target_model, autoencoder, prompt, target_feature, multiplier=5, top_k=10):
    
    model_mlp_out = target_model(prompt, stop_at_layer=0)
    print('model_mlp_out.shape', model_mlp_out.shape)
    hidden_activations = model_mlp_out @ autoencoder.W_enc
    print('hidden_activations.shape', hidden_activations.shape)
    hidden_activations[:, :, target_feature] *= multiplier
    augmented_reconstruction = autoencoder.decode(hidden_activations)
    print('augmented_reconstruction.shape', augmented_reconstruction.shape)

    prompt_length = len(target_model.to_str_tokens(prompt, prepend_bos=True))
    logits = target_model(augmented_reconstruction, start_at_layer=1)
    logits = utils.remove_batch_dim(logits)
    token_probs = logits.softmax(dim=-1)
    token_probs = token_probs[prompt_length-1]
    sorted_token_probs, sorted_token_values = token_probs.sort(descending=True)

    for i in range(top_k):
        print(
                f"Top {i}th token. Logit: {logits[prompt_length-1, sorted_token_values[i]].item():5.2f} Prob: {sorted_token_probs[i].item():6.2%} Token: |{target_model.to_string(sorted_token_values[i])}|"
                )

    return sorted_token_values


steered_output = steered_forward(model, encoder, prompt, 7, 100)