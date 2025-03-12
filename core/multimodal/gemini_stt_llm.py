import asyncio
import os
from google import genai
import contextlib

import wave

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"), http_options={'api_version': 'v1alpha'})
model_id = "gemini-2.0-flash-exp"
config = {"response_modalities": ["AUDIO"]}

@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf

async def main():
    async with client.aio.live.connect(model=model_id, config=config) as session:
        wav_file = wave_file("output.wav")
        while True:
            message = input("User> ")
            if message.lower() == "exit":
                break
            await session.send(input=message, end_of_turn=True)

            with wave_file("output.wav") as wav:
                async for response in session.receive():
                    if response.data is not None:
                        wav.writeframes(response.data)

if __name__ == "__main__":
    asyncio.run(main())