from openai import OpenAI
from dotenv import load_dotenv
import os
import logging


class DeepSeekApi:
  def __init__(self):
    # Obtain API key
    load_dotenv()
    self.api_key = os.getenv("DEEPSEEK_API_KEY")

  # Function to call DeepSeek API to rephrase text
  def rephrase(self, text):
    prompt = f"rephrase: {text}"
    return self._call_api(prompt)
  
  # Function to call DeepSeek API to classify if text is PCL or not (baseline)
  def pcl_classify(self, text):
    prompt = f"Accoording to the paper \"Don’t Patronize Me! An Annotated Dataset with Patronizing and Condescending Language towards Vulnerable Communities\", please classify this sentence \"{text}\" on wheter it is considered a Patronizing and Condescendig Language (PCL) or not. Just reply me with either \"True\" if you think this is ccnsiderd PCL, or \"False\" if you think otherwise."
    return self._call_api(prompt)

  # Function to call DeepSeek API to classify if text is PCL or not in batches (baseline)
  def batch_pcl_classify(self, texts, batch_size):
    prompts = []
    for text in texts:
      prompt = f"Accoording to the paper \"Don’t Patronize Me! An Annotated Dataset with Patronizing and Condescending Language towards Vulnerable Communities\", please classify this sentence \"{text}\" on wheter it is considered a Patronizing and Condescendig Language (PCL) or not. Just reply me with either \"True\" if you think this is ccnsiderd PCL, or \"False\" if you think otherwise."
      prompts.append(prompt)
    return self._batch_call_api(prompts, batch_size)
  
  def _call_api(self, prompt):
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    return response.choices[0].message.content

  def _batch_call_api(self, prompts, batch_size):
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
    
    messages = []
    for prompt in prompts:
      messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False
    )
    
    logging.info(f"Original: {response.choices[0].message.content}")
    predictions = response.choices[0].message.content.split(' \n')
    logging.info(f"After split: {predictions}")
    predictions = list(map(lambda s: "".join(s.split()), predictions))
    logging.info(f"After remove space: {predictions}")
    if '.' in predictions[0]:
      predictions = list(map(lambda s: "".join(s.split('.')[1:]), predictions))
      logging.info(f"After remove dot: {predictions}")
    # Filter out the empty strings
    predictions = [s for s in predictions if s != '']
    logging.info(f"After remove empty str: {predictions}")
    
    logging.info(predictions)
    
    if len(predictions) != batch_size:
      raise ValueError("Error processing predictions from DeepSeek response.")

    return predictions



