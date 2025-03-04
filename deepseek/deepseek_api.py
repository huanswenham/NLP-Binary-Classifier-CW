from openai import OpenAI
from dotenv import load_dotenv
import os


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

