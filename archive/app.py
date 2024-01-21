from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import requests
import os
from pydub import AudioSegment

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# img2text
def img2text(url): # url of image file
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    text = image_to_text(url,max_new_tokens=100)[0]["generated_text"]
    
    print(text)
    return text 


# llm: to generate short story based on the image

def generate_story(scenario, max_length=250, temperature=0.9):
    API_URL = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    
    prompt = scenario
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": max_length,
            "temperature": temperature
        }
    }

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    story = query(payload)[0]["generated_text"]
    return story

# # Example usage
# scenario = "A mysterious island appears overnight in the middle of the ocean."
# story = generate_story(scenario)
# print(story)
    


# import requests

# API_URL = "https://api-inference.huggingface.co/models/lysandre/text-to-speech-pipeline"
# headers = {"Authorization": "Bearer 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"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.content

# audio_bytes = query({
# 	"inputs": "liquid drum and bass, atmospheric synths, airy sounds",
# })
# # You can access the audio with IPython.display for example
# from IPython.display import Audio
# Audio(audio_bytes)

def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization":f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
        "inputs": message
    }
    
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)
        
# ## Testing img2text
scenario = img2text("playful_dogs.jpeg")
# Testing generate_story
story = generate_story(scenario, max_length = 500, temperature= 0.8)[:-21]
print(story)

text2speech(story)

flac_audio = AudioSegment.from_file("audio.flac", "flac")
flac_audio.export("audiostory.mp3", format="mp3")