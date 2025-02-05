'''
This is the subclass of the BaseLLM class.
It uses the OpenAI Chat API to generate responses.
'''

import openai
from core.llm.base_llm import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(self, model="gpt-4-1106-preview"):
        self.model = model
        self.history = []
        
    def generate_response(self, prompt):
        self.history.append({"role": "user", "content": prompt})
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an HR interviewer..."},
                *self.history
            ]
        )
        
        return response.choices[0].message.content
