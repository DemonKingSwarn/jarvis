from ollama import chat
from kokoro import KPipeline
from fzf import fzf_prompt
import soundfile as sf
import numpy as np

import subprocess as sp

pipeline = KPipeline(lang_code='a')

ollama_models = sp.run(['ollama', 'list'], capture_output=True, text=True)
lines = ollama_models.stdout.splitlines()[1:]

selected_model = fzf_prompt(lines)

print(selected_model)

def chat_with_me():
    answer = ""
    while True:
        query = input("You: ")
   
        stream = chat(
            model = "qwen2.5:3b",
            messages = [{'role': 'user', 'content': query}],
            stream = True,
        )
    
        for chunk in stream:
            answer += chunk['message']['content']
    
        print(answer)
        print("\n")
        say(answer)
        answer = ""

def say(text):
    
    generator = pipeline(
        text, voice='af_heart',
        speed=1, split_pattern=r'\n+'
    )
    
    all_audio = []

    for i, (gs, ps, audio) in enumerate(generator):
        all_audio.append(audio)

    concatenated_audio = np.concatenate(all_audio)
    sf.write('output.wav', concatenated_audio, 24000)
    sp.call("paplay output.wav", shell=True)

chat_with_me()
