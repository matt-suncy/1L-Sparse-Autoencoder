import torch
import pandas as pd
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))
csv_files = [os.path.join(dir_path, f'responses_rank_{rank}.csv') for rank in range(torch.cuda.device_count())]
test_df = pd.read_csv(csv_files[0], header=0)
print(test_df.head())
# df_concat = pd.concat([pd.read_csv(f) for f in csv_files ], ignore_index=True, axis=1)
# df_concat.to_csv('responses.csv', encoding='utf-8', index=False, header=True)