# general imports
import os
import sys
import torch
from tqdm import tqdm
import plotly.express as px

torch.set_grad_enabled(False);

import csv
import copy
import pandas as pd
# import itertools as ittl
# import numpy as np
# import math
# from scipy import stats
# from scipy.spatial import distance
# from sklearn import metrics
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator
from datasets import Dataset
from datasets import load_dataset, concatenate_datasets
# import transformers

# package import
from torch import Tensor
from transformer_lens import utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial
from jaxtyping import Int, Float


# device setup
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

from transformer_lens import HookedTransformer
from sae_lens import SAE
import sae_lens.toolkit.pretrained_saes as pretrained_saes

import torch.distributed as dist
import torch.multiprocessing as mp

# Global variables for parameters

### WILL CHANGE
steering_on = False
steer_type = 'ablate'
feature_id = 0
layer = 0
sae, cfg_dict, _ = SAE.from_pretrained(
        release = "gpt2-small-res-jb",
        sae_id = f"blocks.{layer}.hook_resid_pre",
        device = device
        )
hook_point = sae.cfg.hook_name

### PERSISTENT
sampling_kwargs = dict(temperature=0.5, top_p=0.1, freq_penalty=5.0)

def steering_hook(resid_pre, hook):

    if resid_pre.shape[1] == 1:
        return

    position = sae.W_dec.shape[-1]
    
    if steering_on:
      steering_vector = sae.W_dec[feature_id]
      resid_pre[:, :position - 1, :] += coeff * steering_vector

def ablation_hook(resid_pre, hook):

    if resid_pre.shape[1] == 1:
        return
    
    if steering_on:

        position = sae.W_dec.shape[-1]
    
        act_strength = (resid_pre @ sae.W_enc)[:, :position-1, feature_id]
        act_strength = torch.unsqueeze(act_strength, -1)
        new_shape = list(act_strength.shape)
        new_shape[-1] = steering_vector.shape[-1]
        act_strength.expand(new_shape)

        resid_pre[:, :position-1, :] -= act_strength * steering_vector

def hooked_generate(prompt_batch, fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        torch.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        result = model.generate(
            stop_at_eos=False,  # avoids a bug on MPS
            input=tokenized,
            max_new_tokens=20,
            do_sample=True,
            **kwargs)
            
    return result



def load_emotion_data(n_per_label: int=5):

    emotion_dataset = load_dataset('dair-ai/emotion', split='train', trust_remote_code=True)
    emotion_dataset = emotion_dataset.shuffle(seed=42)

    parted_dataset = [emotion_dataset.filter(lambda x: x['label'] == label) for label in range(len(emotion_dataset.features['label'].names))]

    batch = concatenate_datasets([part.select([x for x in range(n_per_label)]) for part in parted_dataset])
    prompts = [f'Alex says, \"{text}\". Alex feels' for text in batch['text']]
    labels = [label for label in batch['label']]
    prompts = list(zip(prompts, labels))

    return prompts




def run_inference(rank, world_size, model, prompts):
    global OUTPUTS

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model.to(f'cuda:{rank}')
    
    # Distribute prompts based on rank
    prompt = prompts[rank::world_size]

    responses = []
    for p, _ in prompt:
        tokenized_p = model.to_tokens(p)
        response = model.generate(max_new_tokens=20, stop_at_eos=False, input=tokenized_p, do_sample=True, **sampling_kwargs)
        response = model.to_string(response[:, 1:])
        response = [(text.split('\". '))[1] for text in response]
        responses.append(response)

    # Collect and save results
    with open(f'responses_rank_{rank}.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for response in responses:
            # TODO: here we can add extra information like orignal of the prompt or whateva
            for r in response:
                spamwriter.writerow([r])

    # if torch.distributed.get_rank() == 0:
    #     prompt = next(prompts)
    
    # elif torch.distributed.get_rank() == 1:
    #     prompt = next(prompts)

    # response = model.generate(prompt, max_new_tokens=10)

    # f = open("parallel_inference_test.txt", "a")
    # f.write(response + '\n')
    # f.close()

def main():

    world_size = torch.cuda.device_count()
    
    model = HookedTransformer.from_pretrained("gpt2-small", device='cpu')

    PROMPTS = load_emotion_data(n_per_label=1)

    mp.spawn(run_inference, args=(world_size, model, PROMPTS), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()

    # Collect responses from all ranks
    csv_files = [f'responses_rank_{rank}.csv' for rank in range(torch.cuda.device_count())]
    df_concat = pd.concat([pd.read_csv(f) for f in csv_files ], ignore_index=True, axis=1)
    df_concat.to_csv('responses.csv', encoding='utf-8', index=False, header=True)
    
    # python -m torch.distributed.run parallel_inference_script.py --nproc_per_node=2



