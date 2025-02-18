from ollama import chat
from kokoro import KPipeline
import soundfile as sf
import numpy as np

import subprocess as sp

pipeline = KPipeline(lang_code='a')

def chat_with_me():
    answer = ""
    while True:
        query = input("You: ")
        
        """
        query = '''
        this prophecy is from percy jackson books

        A half-blood of the eldest gods
        Shall reach sixteen against all odds
        And see the world in endless sleep
        The hero's soul, cursed blade shall reap
        A single choice shall end his days
        Olympus to preserve or raze
        '''
        """
    
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
