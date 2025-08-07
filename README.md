# Real-Time Conversation Assistant

A Flask-based application that provides real-time conversation assistance using OpenAI's Whisper and GPT-4 APIs. The app listens to audio input, transcribes it, and provides AI-generated conversation suggestions.

## Features

- 🎤 Real-time audio processing
- 📝 Live transcription using OpenAI Whisper
- 💡 AI-powered conversation suggestions
- 🔊 Text-to-speech audio responses
- 🌐 WebSocket-based real-time communication
- 🎨 Modern, responsive UI

## Quick Deployment Options

### 1. Railway (Recommended - Easiest)

1. **Fork/Clone this repository**
2. **Sign up at [Railway](https://railway.app)**
3. **Connect your GitHub repository**
4. **Add Environment Variables:**
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `SECRET_KEY`: A random secret key for Flask
5. **Deploy!** Railway will automatically detect it's a Python app and deploy it.

### 2. Render

1. **Sign up at [Render](https://render.com)**
2. **Create a new Web Service**
3. **Connect your GitHub repository**
4. **Configure:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
5. **Add Environment Variables:**
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `SECRET_KEY`: A random secret key for Flask
6. **Deploy!**

### 3. Heroku

1. **Install Heroku CLI**
2. **Create a Heroku account**
3. **Run these commands:**
   ```bash
   heroku create your-app-name
   heroku config:set OPENAI_API_KEY=your_api_key_here
   heroku config:set SECRET_KEY=your_secret_key_here
   git push heroku main
   ```

## Local Development

### Prerequisites

- Python 3.8+
- OpenAI API key
- Audio input device

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd conversationstest
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   export SECRET_KEY="your_secret_key_here"
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Open your browser and go to:** `http://localhost:5000`

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `SECRET_KEY`: Flask secret key for session management (optional, defaults to 'your-secret-key-here')

## How It Works

1. **Audio Capture**: The app continuously captures audio from your microphone
2. **Noise Reduction**: Audio is processed to reduce background noise
3. **Transcription**: OpenAI Whisper transcribes the audio to text
4. **AI Analysis**: GPT-4 analyzes the conversation and generates 3-word suggestions
5. **Audio Response**: The suggestion is converted to speech and played back
6. **Real-time Updates**: All results are displayed in real-time via WebSocket

## Technical Details

- **Backend**: Flask with Flask-SocketIO
- **Audio Processing**: sounddevice, soundfile, noisereduce
- **AI Services**: OpenAI Whisper (transcription), GPT-4 (suggestions), TTS (speech)
- **Frontend**: HTML5, CSS3, JavaScript with Socket.IO client
- **Real-time Communication**: WebSocket via Socket.IO

## Troubleshooting

### Common Issues

1. **Audio not working**: Make sure your browser has microphone permissions
2. **OpenAI API errors**: Check your API key and billing status
3. **Deployment issues**: Ensure all environment variables are set correctly

### Audio Requirements

- Microphone access in browser
- HTTPS required for microphone access in production
- Supported browsers: Chrome, Firefox, Safari, Edge

## License

This project is open source and available under the MIT License. 