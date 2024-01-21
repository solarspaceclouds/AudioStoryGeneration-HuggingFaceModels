import streamlit as st
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import requests
import os
from pydub import AudioSegment
import re
from transformers import VitsModel, AutoTokenizer
import torch
import scipy
import scipy.io.wavfile
import torch
import soundfile as sf

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# img2text
def img2text(url): # url of image file
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    
    text = image_to_text(url,max_new_tokens=100)[0]["generated_text"]
    
    print(text)
    return text 

# llm: to generate short story based on the image
def generate_story(scenario, max_length=500, temperature=0.9):
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

# def text2speech(message, voice_type):
def text2speech(message):
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

    # Select the model based on the voice type
    # model = voice_models[voice_type]
    
    inputs = tokenizer(message, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform
    print(output.float().numpy().shape)
    scipy.io.wavfile.write("audiostory.wav", rate=model.config.sampling_rate, data=output.squeeze().float().numpy())
    flac_audio = AudioSegment.from_wav("audiostory.wav")
    flac_audio.export("audiostory.mp3", format="mp3")
        

def text2speech2(message):
    # Create a pipeline for text-to-speech
    tts_pipeline = pipeline("text-to-speech", model="facebook/mms-tts-eng")

    # Generate the audio waveform
    outputs = tts_pipeline(message)
    if isinstance(outputs, tuple):
        output = outputs[0]  # In case the output is a tuple, take the first element
    else:
        output = outputs

    print("output:",output)
    # Convert the output to a numpy array and squeeze it to 1D
    audio_data = output["audio"].squeeze()

    # Write to a WAV file
    scipy.io.wavfile.write("audiostory.wav", rate=24000, data=audio_data)  # Replace 24000 with actual sample rate if different

    # Convert to MP3 using pydub
    AudioSegment.from_wav("audiostory.wav").export("audiostory.mp3", format="mp3")
    

# TRY VOICE CLONING
# def text2speech3(message, speaker_audio_file='female_voice.wav'): 
#     # Load the target speaker's audio file
#     speaker_audio, _ = sf.read(speaker_audio_file)
    
#     # Initialize the text-to-audio pipeline with the specified model
#     pipe = pipeline("text-to-audio", model="ashleyliu31/ashley_voice_clone_speecht5_finetuning")

#     # Generate the audio waveform with the text and speaker audio
#     output = pipe({"text": message, "speech": speaker_audio})

#     # The output is typically a dictionary with keys 'sampling_rate' and 'audio'
#     audio_data = output['audio'].squeeze()
#     sampling_rate = output['sampling_rate']
    
#     # Write to a WAV file
#     scipy.io.wavfile.write("audiostory.wav", rate=sampling_rate, data=audio_data)  # Replace 24000 with actual sample rate if different

#     # Convert to MP3 using pydub
#     AudioSegment.from_wav("audiostory.wav").export("audiostory.mp3", format="mp3")
    
def get_last_complete_sentence(text):
    # Split the text by period and get the last complete sentence
    sentences = re.split(r'(?<=[.!?]) +', text)
    if sentences:
        return [' '.join(sentences[:-1]),sentences[:-1]]  # Exclude the last incomplete sentence
    return text, sentences # 

def main():
    st.title("Image to Story App")

    # Input for image URL
    url = st.text_input("Enter the URL of an image:")
    
    #  # Voice selection
    # voice_type = st.radio("Choose a voice for the narration:", ('male', 'female'))

    # Process image
    if st.button("Generate Story from Image"):
        if url:
            # Display the image
            st.markdown("### Chosen Image")
            st.image(url, caption="Selected Image", use_column_width=True)

            with st.spinner('Generating caption from image...'):
                scenario = img2text(url)
            st.markdown("### Image Caption")
            st.write(scenario)

            st.markdown("### Generated Story")
            st.write("Using the above caption as a prompt for the story:")
            with st.spinner('Generating story...'):
                story = generate_story(scenario, max_length=500, temperature=0.9)
                story, story_list = get_last_complete_sentence(story)
            st.write(story)

            st.markdown("### Audio of the Story")
            with st.spinner('Converting story to speech...'):
                # text2speech(story, voice_type)
                text2speech(story)
                # text2speech2(story)
                # text2speech3(story_list)
                
                # Display audio player and download button
                audio_file = open("audiostory.mp3", "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
                audio_file.seek(0)
                st.download_button(
                    label="Download Story as MP3",
                    data=audio_file,
                    file_name="audiostory.mp3",
                    mime="audio/mp3"
                )
        else:
            st.warning("Please enter a URL.")

if __name__ == "__main__":
    main()