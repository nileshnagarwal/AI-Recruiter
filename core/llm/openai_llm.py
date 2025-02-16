'''
This is the subclass of the BaseLLM class.
It uses the OpenAI Chat API to generate responses.
'''

import openai
import os
import json
from dotenv import load_dotenv
load_dotenv(dotenv_path='../../.env')

openai.api_key = os.getenv("OPENAI_API_KEY")
from core.llm.base_llm import BaseLLM

class OpenAILLM(BaseLLM):
    def __init__(self, model="gpt-4o-mini-2024-07-18"):
        self.model = model
        self.history = []
        self.system_prompt = """You are an HR interviewer conducting a job interview. Be professional and thorough in your questions and responses.
If the candidate indicates they want to end the interview (by saying goodbye, thank you, or similar phrases), respond appropriately and set should_exit=true in your response.
Format your response as a JSON object with two fields:
{
    "response": "Your actual response text",
    "should_exit": boolean indicating if the conversation should end
}"""
        
    def generate_response(self, prompt, stream=False):
        """Generate a response from the LLM.
        
        Args:
            prompt: The user's input text
            stream: Whether to stream the response
            
        Returns:
            If stream=False: A dict with 'response' and 'should_exit' fields
            If stream=True: A generator yielding sentence fragments
        """
        self.history.append({"role": "user", "content": prompt})
        
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                *self.history
            ],
            response_format={ "type": "json_object" },
            stream=stream
        )
        
        if not stream:
            response_json = eval(response.choices[0].message.content)
            self.history.append({"role": "assistant", "content": response_json["response"]})
            return response_json
        else:
            # For streaming, we need to accumulate the JSON response
            accumulated_text = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    accumulated_text += chunk.choices[0].delta.content
                    try:
                        # Try to parse the accumulated text as JSON
                        response_json = json.loads(accumulated_text)
                        # If successful, yield the next part of the response
                        yield response_json["response"]
                        break
                    except json.JSONDecodeError:
                        # If not valid JSON yet, continue accumulating
                        continue
            
            # Save the complete response to history
            self.history.append({"role": "assistant", "content": response_json["response"]})
            return response_json
