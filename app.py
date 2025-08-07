from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
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
import threading
import queue
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# === CONFIGURATION ===
CHUNK_DURATION = 5         # seconds
STEP_DURATION = 1          # seconds (overlap rate)
SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
GAIN = 2.5                 # Amplify signal
TARGET_PEAK = 0.6          # AGC target volume
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
BUFFER_SIZE = int(CHUNK_DURATION * SAMPLE_RATE)

# === GLOBAL STATE ===
buffer = np.zeros(BUFFER_SIZE, dtype='int16')
buffer_pos = 0
is_listening = False
audio_stream = None

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

# === MAIN PROCESSING LOOP ===
def processing_loop():
    global is_listening
    while is_listening:
        try:
            audio_path = get_last_5_seconds()
            # Run async functions in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            transcript = loop.run_until_complete(transcribe_api(audio_path))
            suggestion = loop.run_until_complete(generate_suggestions(transcript))
            
            # Emit results to frontend
            socketio.emit('transcript', {'text': transcript.strip()})
            socketio.emit('suggestion', {'text': suggestion.strip()})
            
            # Generate and emit audio
            tts_audio_path = loop.run_until_complete(text_to_speech(suggestion))
            
            # Read audio file and send as base64
            with open(tts_audio_path, 'rb') as f:
                import base64
                audio_data = base64.b64encode(f.read()).decode('utf-8')
                socketio.emit('audio_response', {'audio': audio_data})
            
            loop.close()
            os.remove(audio_path)
            os.remove(tts_audio_path)
            time.sleep(STEP_DURATION)
        except Exception as e:
            print(f"[⚠️ ERROR] {e}")
            socketio.emit('error', {'message': str(e)})
            time.sleep(STEP_DURATION)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_listening')
def handle_start_listening():
    global is_listening, audio_stream
    if not is_listening:
        is_listening = True
        audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=audio_callback
        )
        audio_stream.start()
        
        # Start processing in separate thread
        processing_thread = threading.Thread(target=processing_loop)
        processing_thread.daemon = True
        processing_thread.start()
        
        emit('status', {'message': 'Started listening'})

@socketio.on('stop_listening')
def handle_stop_listening():
    global is_listening, audio_stream
    is_listening = False
    if audio_stream:
        audio_stream.stop()
        audio_stream.close()
        audio_stream = None
    emit('status', {'message': 'Stopped listening'})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True) 