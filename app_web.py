from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import tempfile
import numpy as np
from openai import OpenAI
import asyncio
import time
import os
import threading
import json
import base64
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# === CONFIGURATION ===
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    print("⚠️ WARNING: OPENAI_API_KEY environment variable not set!")
    print("Please set the OPENAI_API_KEY environment variable in Railway.")
    print("The app will start but won't be able to process audio.")
    client = None
else:
    client = OpenAI(api_key=api_key)

# === GLOBAL STATE ===
conversation_history = []  # Track recent conversation topics

# === TRANSCRIBE WITH WHISPER ===
async def transcribe_api(audio_data):
    print("🧠 Transcribing...")
    # Create temporary file from base64 audio data
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.write(base64.b64decode(audio_data))
    temp_file.close()
    
    with open(temp_file.name, "rb") as f:
        transcription = await asyncio.to_thread(
            client.audio.transcriptions.create, model="whisper-1", file=f
        )
    
    os.remove(temp_file.name)
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

@socketio.on('process_audio')
def handle_process_audio(data):
    """Handle audio data sent from the frontend"""
    try:
        if not client:
            emit('error', {'message': 'OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.'})
            return
            
        audio_data = data.get('audio')
        if not audio_data:
            emit('error', {'message': 'No audio data received'})
            return
        
        # Run async functions in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        transcript = loop.run_until_complete(transcribe_api(audio_data))
        suggestion = loop.run_until_complete(generate_suggestions(transcript))
        
        # Emit results to frontend
        emit('transcript', {'text': transcript.strip()})
        emit('suggestion', {'text': suggestion.strip()})
        
        # Generate and emit audio
        tts_audio_path = loop.run_until_complete(text_to_speech(suggestion))
        
        # Read audio file and send as base64
        with open(tts_audio_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
            emit('audio_response', {'audio': audio_data})
        
        loop.close()
        os.remove(tts_audio_path)
        
    except Exception as e:
        print(f"[⚠️ ERROR] {e}")
        emit('error', {'message': str(e)})

@socketio.on('start_listening')
def handle_start_listening():
    global conversation_history
    conversation_history = []  # Reset conversation history
    emit('status', {'message': 'Started listening - send audio data to process_audio event'})

@socketio.on('stop_listening')
def handle_stop_listening():
    emit('status', {'message': 'Stopped listening'})

if __name__ == '__main__':
    print("🎤 Heygent A.I. - Web Version")
    print("🌐 Open your browser and go to: http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True) 