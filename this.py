#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
sys.path.append('/home/dividivi/flask_dir/')


# In[4]:


import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import dataloader
from dialogLM.dataloader.wellness import WellnessAutoRegressiveDataset
from dialogLM.model.kogpt2 import DialogKoGPT2


# In[ ]:


import os
import numpy as np
import torch
from dialogLM.model.kogpt2 import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer

root_path='/home/dividivi/flask_dir/dialogLM'
data_path = f"{root_path}/data/wellness_dialog_for_autoregressive_train.txt"
checkpoint_path =f"{root_path}/checkpoint"
# save_ckpt_path = f"{checkpoint_path}/kogpt2-wellnesee-auto-regressive.pth"
save_ckpt_path = f"{checkpoint_path}/kogpt2-catbot-wellness0.pth"

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

# 저장한 Checkpoint 불러오기
checkpoint = torch.load(save_ckpt_path, map_location=device)

model = DialogKoGPT2()
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

tokenizer = get_kogpt2_tokenizer()

count = 0
output_size = 200 # 출력하고자 하는 토큰 갯수

while 1:
# for i in range(5):
  sent = input('Question: ')  # '요즘 기분이 우울한 느낌이에요'
  if sent.encode().isalpha():
    print("한국말로 해주세용^^")
    print(100 * '-')
    continue
  tokenized_indexs = tokenizer.encode(sent)

  input_ids = torch.tensor([tokenizer.bos_token_id,]  + tokenized_indexs +[tokenizer.eos_token_id]).unsqueeze(0)
  # set top_k to 50
  sample_output = model.generate(input_ids=input_ids)


  str= tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs)+1:],skip_special_tokens=True)
  print("Answer: " +str.split('.')[0])
  print(100 * '-')

