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

def steered_forward(target_model, autoencoder, prompt, target_feature, multiplier=5):
    model_mlp_out = target_model(prompt, output_hidden_states=True).hidden_states[-1]
    print('model_mlp_out.shape', model_mlp_out.shape)
    print('model_mlp_out', model_mlp_out)
    hidden_activations = autoencoder.encode(model_mlp_out)
    print('hidden_activations.shape', hidden_activations.shape)
    print('hidden_activations', hidden_activations)
    hidden_activations[0][target_feature] = hidden_activations[0][target_feature] * multiplier
    print('hidden_activations', hidden_activations)
    augmented_reconstruction = autoencoder.decode(hidden_activations)
    print('augmented_reconstruction.shape', augmented_reconstruction.shape)
    print('augmented_reconstruction', augmented_reconstruction)
    output = target_model(augmented_reconstruction, output_hidden_states=True)
    return output
