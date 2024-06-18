import huggingface_hub
from pathlib import Path
import os 
import wandb
import shutil

# Set your Hugging Face token as an environment variable
os.environ['HUGGINGFACE_HUB_TOKEN'] = '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'

'''
def push_to_hub(local_dir):
    if isinstance(local_dir, huggingface_hub.Repository):
        local_dir = local_dir.local_dir
    os.system(f"git -C {local_dir} add .")
    os.system(f"git -C {local_dir} commit -m 'Auto Commit'")
    os.system(f"git -C {local_dir} push")
'''

def push_to_hub(local_dir):
    repo.git_add()
    repo.git_commit("Auto Commit")
    repo.git_push()

def upload_folder_to_hf(folder_path, repo_name=None, debug=False):
    folder_path = Path(folder_path)
    if repo_name is None:
        repo_name = folder_path.name
    repo_folder = folder_path.parent / (folder_path.name + "_repo")

    token = "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"
    huggingface_hub.login(token, add_to_git_credential=True)

    repo_url = huggingface_hub.create_repo(repo_name, exist_ok=True)
    repo = huggingface_hub.Repository(repo_folder, repo_url)

    # Set the pull strategy to merge
    os.system(f'git -C {repo_folder} config pull.rebase false')

    repo.git_pull()

    for file in folder_path.iterdir():
        if debug:
            print(file.name)
        shutil.move(str(file), repo_folder / file.name)

    push_to_hub(repo.local_dir)

path = r'/home/jovyan/1L-Sparse-Autoencoder/checkpoint_storage'
#upload_folder_to_hf(path, "sparse_autoencoder", True)
