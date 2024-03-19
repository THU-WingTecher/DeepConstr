


import os
from dotenv import load_dotenv

def proxy_check(cfg) : 
    from neuri.constrinf.inferencer import Inferencer
    inferencer = Inferencer(cfg['llm']['settings'])
    inferencer.change_to_gpt3()
    prompts = 'hello'
    results = inferencer.inference(prompts)
    print(results)

def check_internet_connection(url='http://www.baidu.com'):
    import requests
    try:
        print(url)
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an error for bad status codes
        print("Internet is connected.")
    except requests.RequestException as e:
        print("Internet is not connected:", e)

import hydra
from omegaconf import DictConfig
@hydra.main(version_base=None, config_path="../neuri/config", config_name="main")
def main(cfg: DictConfig):
    load_dotenv(override=True)
    os.environ['ALL_PROXY'] =os.getenv("MYPROXY")
    check_internet_connection("http://www.google.com")
    proxy_check(cfg)

if __name__ == "__main__" :
    main()