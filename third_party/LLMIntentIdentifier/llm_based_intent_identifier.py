import hydra
import re
import requests
from prompts.intent_identifier.instruction import intent_identifier_instruction_prompt
from prompts.intent_identifier.few_shot_examples import intent_identifier_few_shot_examples

class IntentIdentifier:
    def __init__(self, cfg):
        self.instruction_prompt = intent_identifier_instruction_prompt
        self.few_shot_examples = intent_identifier_few_shot_examples
        self.example_num = cfg.example_num
        self.llm_address = cfg.llm_address
    
    def predict(self, question):
        # generate prompt
        prompt = self.prompt_generate(question)
        
        # send request to llm
        response_list = requests.post(
                self.llm_address,
                json={
                    "text": [prompt],
                    "sampling_params": {
                        "max_new_tokens": 40,
                        "temperature": 0,
                    }
                },
                timeout=None,
            ).json()
        
        # parse response
        pattern_col = r"f_tune\(\[(.*?)\]\)"

        try:
            pred = re.findall(pattern_col, response_list[0]["text"], re.S)[0].strip()
        except:
            return False
        
        if 'true' == pred.lower():
            return True
        else:
            return False

    def prompt_generate(self, question):
        return self.instruction_prompt.format(question=question, few_shot_examples='\n\n'.join(self.few_shot_examples[:self.example_num]))
    
@hydra.main(config_path="conf", config_name="intent_identifier")
def main(cfg):
    intent_identifier = IntentIdentifier(cfg)
    print(intent_identifier.predict("What is the capital of France?"))

if __name__ == "__main__":
    main()