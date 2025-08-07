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
import threading
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# === CONFIGURATION ===
CHUNK_DURATION = 5         # seconds
STEP_DURATION = 2          # seconds (overlap rate) - increased from 1
SAMPLE_RATE = 16000
GAIN = 2.5                 # Amplify signal
TARGET_PEAK = 0.6          # AGC target volume
SILENCE_THRESHOLD = 0.02   # Minimum audio level to consider as speech - increased from 0.01
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
BUFFER_SIZE = int(CHUNK_DURATION * SAMPLE_RATE)

# === GLOBAL STATE ===
buffer = np.zeros(BUFFER_SIZE, dtype='int16')
buffer_pos = 0
is_listening = False
audio_stream = None
last_processed_time = 0
MIN_PROCESSING_INTERVAL = 5  # Minimum seconds between processing - increased from 3
conversation_history = []  # Track recent conversation topics

# === AUDIO CALLBACK ===
def audio_callback(indata, frames, time, status):
    global buffer, buffer_pos
    if status:
        print(status)

    data = indata[:, 0].astype(np.float32)

    # Simple amplification
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

# === CHECK IF AUDIO HAS SPEECH ===
def has_speech(audio_data):
    """Check if audio contains speech by measuring RMS level and peak amplitude"""
    # Convert to float32 for calculations
    audio_float = audio_data.astype(np.float32) / 32767.0
    
    # Calculate RMS (Root Mean Square) - overall volume
    rms = np.sqrt(np.mean(audio_float**2))
    
    # Calculate peak amplitude
    peak = np.max(np.abs(audio_float))
    
    # Check if audio has sufficient volume and dynamic range
    has_volume = rms > SILENCE_THRESHOLD
    has_dynamics = peak > (SILENCE_THRESHOLD * 3)  # Need some dynamic range
    
    return has_volume and has_dynamics

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
    global conversation_history
    print("🤖 Generating conversation directions...")
    
    # Add current transcript to history (keep last 3)
    conversation_history.append(transcript.strip())
    if len(conversation_history) > 3:
        conversation_history = conversation_history[-3:]
    
    # Create context from recent conversation
    context = " | ".join(conversation_history[-2:]) if len(conversation_history) > 1 else transcript.strip()
    
    prompt = f"""
You are a conversation coach who helps guide natural, engaging conversations. Based on the recent conversation context, provide a 3-word phrase that would naturally lead the conversation forward in an interesting direction.

Recent conversation context: "{context}"

Your response should:
- Feel natural and conversational
- Build on what was just discussed
- Open up new topics or deeper exploration
- Be specific and actionable
- Avoid generic responses
- Create opportunities for meaningful dialogue

Examples of good responses:
- "Tell me more" (when someone mentions something briefly)
- "What happened next" (for stories)
- "How did that feel" (for experiences)
- "What's your take" (for opinions)
- "Let's explore that" (for interesting topics)

Return only the 3-word phrase, nothing else. Make it feel like a natural next step in the conversation.
"""
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
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
    global is_listening, last_processed_time
    while is_listening:
        try:
            current_time = time.time()
            
            # Check if enough time has passed since last processing
            if current_time - last_processed_time < MIN_PROCESSING_INTERVAL:
                time.sleep(0.5)
                continue
            
            # Check if audio contains speech
            if not has_speech(buffer):
                time.sleep(0.5)
                continue
            
            audio_path = get_last_5_seconds()
            
            # Run async functions in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            transcript = loop.run_until_complete(transcribe_api(audio_path))
            
            # Skip if transcript is too short or contains common background phrases
            transcript_clean = transcript.strip().lower()
            skip_phrases = [
                "thank you for watching", "background", "you", "sex", "uh", "um", "ah",
                "thank you", "watching", "video", "subscribe", "like", "comment",
                "background music", "background noise", "silence", "quiet"
            ]
            
            # Check if transcript should be skipped
            should_skip = (
                len(transcript_clean) < 5 or  # Too short
                any(phrase in transcript_clean for phrase in skip_phrases) or
                transcript_clean in ["you", "sex", "uh", "um", "ah", "hmm", "yeah", "no"]
            )
            
            if should_skip:
                print(f"⏭️ Skipping: '{transcript.strip()}' (background noise)")
                loop.close()
                os.remove(audio_path)
                time.sleep(STEP_DURATION)
                continue
            
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
            
            last_processed_time = current_time
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
    global is_listening, audio_stream, last_processed_time, conversation_history
    if not is_listening:
        is_listening = True
        last_processed_time = 0
        conversation_history = []  # Reset conversation history
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
    print("🎤 Heygent A.I.")
    print("🌐 Open your browser and go to: http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True) 