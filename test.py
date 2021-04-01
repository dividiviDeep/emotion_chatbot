import os
import json
import numpy as np
import torch
from dialogLM.model.kogpt2 import DialogKoGPT2
from kogpt2_transformers import get_kogpt2_tokenizer
from flask import Flask,request,Response,render_template

arr=[]


app=Flask(__name__)


@app.route("/",methods=['POST','GET'])
def hello():
    return render_template('test.html')


@app.route("/post",methods=['POST','GET'])
def home():
    root_path='..'
    data_path = f"{root_path}/data/wellness_dialog_for_autoregressive_train.txt"
    checkpoint_path =f"{root_path}/checkpoint"
    # save_ckpt_path = f"{checkpoint_path}/kogpt2-wellnesee-auto-regressive.pth"
    save_ckpt_path = f"/home/dividivi/flask_dir/dialogLM/checkpoint/kogpt2-catbot-wellness0.pth"
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
      sent = request.form['msg']  # '요즘 기분이 우울한 느낌이에요'
      if sent.encode().isalpha():
        print("한국말로 해주세용^^")
        print(100 * '-')
        continue
      tokenized_indexs = tokenizer.encode(sent)

      input_ids = torch.tensor([tokenizer.bos_token_id,]  + tokenized_indexs +[tokenizer.eos_token_id]).unsqueeze(0)
      # set top_k to 50
      sample_output = model.generate(input_ids=input_ids)

      str= tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs)+1:],skip_special_tokens=True)
      answer=str.split('.')[0]
      arr.append(sent)
      arr.append(answer)
    
      return render_template('test.html',msg=arr)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
