import sounddevice as sd
import soundfile as sf
import tempfile
import numpy as np
from openai import OpenAI
import asyncio
import time
import os
import noisereduce as nr
from scipy import signal

# === CONFIGURATION ===
CHUNK_DURATION = 5         # seconds
STEP_DURATION = 1          # seconds (overlap rate)
SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
GAIN = 2.5                 # Amplify signal
TARGET_PEAK = 0.6          # AGC target volume
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
BUFFER_SIZE = int(CHUNK_DURATION * SAMPLE_RATE)

# === RING BUFFER ===
buffer = np.zeros(BUFFER_SIZE, dtype='int16')
buffer_pos = 0

# === AUDIO CALLBACK ===
def audio_callback(indata, frames, time, status):
    global buffer, buffer_pos
    if status:
        print(status)

    data = indata[:, 0].astype(np.float32)

    # Denoise
    data = nr.reduce_noise(y=data, sr=SAMPLE_RATE)

    # Amplify
    data *= GAIN

    # Automatic Gain Control
    peak = np.max(np.abs(data)) + 1e-6
    data *= TARGET_PEAK / peak

    # Clip to [-1.0, 1.0]
    data = np.clip(data, -1.0, 1.0)

    # Convert back to int16 for buffer
    data_int16 = (data * 32767).astype('int16')

    n = len(data_int16)
    buffer[:-n] = buffer[n:]
    buffer[-n:] = data_int16
    buffer_pos += n

# === EXTRACT LAST 5 SECONDS ===
def get_last_5_seconds():
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, buffer, SAMPLE_RATE)
    return temp_file.name

# === TRANSCRIBE WITH WHISPER ===
async def transcribe_api(audio_path):
    print("🧠 Transcribing...")
    with open(audio_path, "rb") as f:
        transcription = await asyncio.to_thread(
            client.audio.transcriptions.create, model="whisper-1", file=f
        )
    print(transcription.text)
    return transcription.text

# === SUGGEST 3-WORD PROMPT ===
async def generate_suggestions(transcript):
    print("🤖 Generating conversation directions...")
    prompt = f"""
you are a helpful assistant that helps me with my conversation, listen to all people in the conversation and give me a 3-word phrase that feels like a natural nudge to overlap interests

Transcript:
"{transcript.strip()}"

Return only the 3-word phrase, nothing else.
"""
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=20
    )
    return response.choices[0].message.content

# === TEXT-TO-SPEECH ===
async def text_to_speech(text):
    print("🔊 Converting suggestion to audio...")
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    response = await asyncio.to_thread(
        client.audio.speech.create,
        model="tts-1",
        voice="alloy",
        input=text
    )
    with open(temp_audio.name, "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
    return temp_audio.name

# === PLAY AUDIO BACK ===
async def play_audio(audio_path):
    print("🎵 Playing suggestion...")
    data, fs = sf.read(audio_path)
    if fs != SAMPLE_RATE:
        data = signal.resample(data, int(len(data) * SAMPLE_RATE / fs))
    sd.play(data, SAMPLE_RATE)
    sd.wait()
    os.remove(audio_path)

# === MAIN LOOP ===
async def main():
    print("💬 Starting in-ear assistant with max mic sensitivity. Ctrl+C to exit.")
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback
    )
    with stream:
        while True:
            try:
                audio_path = get_last_5_seconds()
                transcript = await transcribe_api(audio_path)
                suggestion = await generate_suggestions(transcript)
                print(f"\n📝 Transcript:\n{transcript.strip()}")
                print(f"\n💡 Conversation Direction:\n{suggestion.strip()}")
                tts_audio_path = await text_to_speech(suggestion)
                await play_audio(tts_audio_path)
                os.remove(audio_path)
                await asyncio.sleep(STEP_DURATION)
            except KeyboardInterrupt:
                print("\n👋 Exiting. Goodbye!")
                break
            except Exception as e:
                print(f"[⚠️ ERROR] {e}")
                await asyncio.sleep(STEP_DURATION)

if __name__ == "__main__":
    asyncio.run(main())
