import random
import time
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple
import os 
import openai
from deepconstr.logger import LLM_LOG
from deepconstr.utils import formatted_dict

load_dotenv(override=True)

class Inferencer() :
    def __init__(self, 
                 setting : Dict[str, Any]) -> None:
        self.setting = setting 
        self.model = None
        self.args = None
        self.update_setting(self.setting)
        self.prompt_token = 0
        self.complete_token = 0
        self.key1 = os.getenv('OPENAI_API_KEY1')
    def init(self) -> None :
        self.prompt_token = 0
        self.complete_token = 0
    def up_temp(self, temp : float) -> None :
        while self.args['temperature']+temp > 2 :
            self.args['temperature'] -= 1
        self.set_temp(self.args['temperature']+temp)
    def down_temp(self, temp : float) -> None :
        while self.args['temperature']-temp < 0 :
            self.args['temperature'] += 1
        self.set_temp(self.args['temperature']-temp)
    def set_temp(self, temp : float) -> None :
        self.args['temperature'] = temp
    def set_model(self, model : str) -> None :
        self.model = model
    def default(self) -> None :
        self.update_setting(self.settings[0])
    def is_gpt4(self) -> bool :
        return self.model == 'gpt-4'
    def change_to_gpt4(self) -> None :
        self.set_model('gpt-4')
        self.set_temp(0.7)
    def change_to_gpt3(self) -> None :
        self.set_model('gpt-3.5-turbo-16k-0613')
        self.set_temp(0.7)
    def update_setting(self, setting : Dict[str, Any]) -> None :
        self.model = setting['model_name']
        self.args = {arg_name : val for arg_name, val in setting.items() if arg_name != 'model_name'}
    def setting_change(self) -> None :
        self.update_setting(random.choice(self.settings))
    def load_prompts(self, prompts : str ) -> None :
        """
        get prompts through argument name 'prompts'
        if 'prompts' is None -> load written prompts from "prompt.txt" in workspace
        """
        self.prompts = prompts
    
    def finalize(self) :
        LLM_LOG.info(
            f"====================\nUsed Prompt : {self.prompt_token} Complete : {self.complete_token} tokens\n====================\n"
        )
        
    def inference(self, 
                  prompts : str, 
                  contexts : str = '', 
                  update : bool = False
                  ) -> str :
        # self.flip_key()
        if os.getenv("MYPROXY") is not None:
            os.environ['ALL_PROXY'] = os.getenv("MYPROXY")
        completion = None

        start = time.time()
        
        LLM_LOG.info(f'Inferencing with {self.model}\{formatted_dict(self.args, split=", ")} \nSystem :\n{contexts}\nPrompts :\n{prompts}\n')
        client = openai.OpenAI(
            api_key=os.getenv('OPENAI_API_KEY1'),
            timeout=self.setting['timeout']
        )
        try:
            completion = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role' : "system", 'content' : contexts},
                        {'role': 'user', 'content': prompts},
                    ],
                    **self.args
            )
        except openai.APIConnectionError as e:
            LLM_LOG.error(e.__cause__)  # an underlying Exception, likely raised within httpx.
        except openai.RateLimitError as e:
            LLM_LOG.error("A 429 status code was received; we should back off a bit.")
        except openai.APIStatusError as e:
            LLM_LOG.error("Another non-200-range status code was received")
            LLM_LOG.error(e.status_code)
            LLM_LOG.error(e.response)   

        time_cost = time.time() - start
        if completion is None : 
            return 
        response = completion.choices[0].message.content
        LLM_LOG.info(f'Output(Ptk{completion.usage.prompt_tokens}-OtkPtk{completion.usage.completion_tokens}) : \n {response} \n Time cost : {time_cost} seconds \n')
        self.prompt_token += completion.usage.prompt_tokens
        self.complete_token += completion.usage.completion_tokens
        return response
    
