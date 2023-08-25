import ctransformers
import time
import requests
from tqdm import tqdm
import os

import uuid

def isCoverPage():
    x = os.environ["REPL_ID"]
    return not (str(uuid.UUID(x, version=4)) == x)

if isCoverPage():
  print("You cannot run this Repl from a cover page, please Fork it to get it running")
  exit()

#Get the model file - you will need Expandable Storage to make this work

if not os.path.isfile('llama-2-7b.ggmlv3.q4_K_S.bin') and not isCoverPage():
  print("Downloading Model from HuggingFace")
  url = "https://huggingface.co/TheBloke/Llama-2-7B-GGML/resolve/main/llama-2-7b.ggmlv3.q4_K_S.bin"
  response = requests.get(url, stream=True)
  total_size_in_bytes= int(response.headers.get('content-length', 0))
  block_size = 1024 #1 Kibibyte
  progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
  with open('llama-2-7b.ggmlv3.q4_K_S.bin', 'wb') as file:
      for data in response.iter_content(block_size):
          progress_bar.update(len(data))
          file.write(data)
  progress_bar.close()
  if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
      print("ERROR, something went wrong")
  
#Sets up the transformer library and adds in the Llama-2 model

configObj = ctransformers.Config(stop=["\n", 'User'])
config = ctransformers.AutoConfig(config=configObj, model_type='llama')
config.config.stop = ["\n"]

llm = ctransformers.AutoModelForCausalLM.from_pretrained('./llama-2-7b.ggmlv3.q4_K_S.bin', config=config)
print("Loaded model")

def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.6f} seconds to execute.")
        return result
    return wrapper

def complete(prompt, stop=["User", "Assistant"]):
  tokens = llm.tokenize(prompt)
  token_count = 0
  output = ''
  for token in llm.generate(tokens):
    token_count += 1
    result = llm.detokenize(token)
    output += result
    for word in stop:
      if word in output:
        print('\n')
        return [output, token_count]
    print(result, end='',flush=True)

  print('\n')
  return [output, token_count]

while True:
  question = input("\nWhat is your question? > ")
  start_time = time.time()
  output, token_count = complete(f'User: {question}\nAssistant: ')
  end_time = time.time()
  execution_time = end_time - start_time
  print(f"{token_count} tokens generated in {execution_time:.6f} seconds.\n{token_count/execution_time} tokens per second")
  
