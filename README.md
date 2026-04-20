# 🎬 AI Video Dubbing & Lip-Sync System

An AI-powered system that automatically translates video speech into different languages and generates lip-synced dubbed videos with preserved background music.

---

## 🚀 Features

- 🎧 Speech-to-text using OpenAI Whisper
- 🌍 Multi-language translation (Tamil, Hindi, Malayalam, etc.)
- 🔊 Neural voice generation using Edge TTS
- 🎬 Lip-sync using Wav2Lip
- 🎼 Background music preservation (Demucs)
- ⏱️ Segment-based timing for natural dubbing
- 📥 Upload & download processed videos via web interface

---

## 🧠 Tech Stack

- **Frontend:** React
- **Backend:** FastAPI
- **AI Models:**
  - Whisper (Speech Recognition)
  - Wav2Lip (Lip Sync)
- **Audio Processing:** FFmpeg, Demucs
- **TTS:** Edge-TTS
- **Translation:** Deep Translator

---

## 📂 Project Structure
ai-video-dubbing/
│
├── backend/
│ ├── main.py
│ ├── wav2lip_runner.py
│ ├── requirements.txt
│ ├── temp/
│ ├── outputs/
│ └── Wav2Lip/
│
├── frontend/
│ ├── src/
│ ├── public/
│ ├── package.json
│
├── .gitignore
└── README.md



---

## ⚙️ Installation & Setup

### 🔹 Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python -m uvicorn main:app --reload
``` 
---

Frontend Setup

```bash
cd frontend
npm install
npm start
```

🎯 How It Works
User uploads a video
Audio is extracted using FFmpeg
Whisper converts speech → text
Text is translated into target language
Edge TTS generates new dubbed voice
Demucs separates background music
New voice + BGM are merged
Wav2Lip syncs lips with generated audio
Final video is returned to user


💡 Use Cases
🎬 Movie dubbing
🌐 Content localization
📺 YouTube video translation
🎤 Voice-over automation


⚠️ Requirements

Python 3.10+

Node.js

FFmpeg installed and added to PATH

GPU recommended for Wav2Lip


🚀 Future Improvements

🎤 Voice cloning (same speaker voice)

⚡ Faster processing pipeline

🌐 Cloud deployment

🎯 Better emotion-aware dubbing


👨‍💻 Author

Adithyan D

⭐ If you like this project, give it a star!
