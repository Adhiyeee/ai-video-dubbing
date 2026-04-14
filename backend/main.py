"""
AI Video Dubbing Pipeline — FINAL VERSION
==========================================
Pipeline:
  1. Whisper (GPU) — transcribe
  2. Segment merge — better translation context
  3. deep-translator — translate
  4. Rule-based grammar fix — natural speech
  5. Edge TTS — neural voice per segment
  6. atempo stretch — fit to segment window
  7. FFmpeg amix — combine segments with timing
  8. FFmpeg merge — dubbed audio into video
  9. Demucs — separate BGM from original voice
 10. Wav2Lip (GPU) — lip sync
 11. Final mix — dubbed voice + pure BGM
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from deep_translator import GoogleTranslator
import shutil, os, subprocess, asyncio, re, time
import torch
import whisper
import edge_tts

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
TEMP_FOLDER   = "temp"
OUTPUT_FOLDER = "outputs"
for d in [UPLOAD_FOLDER, TEMP_FOLDER, OUTPUT_FOLDER]:
    os.makedirs(d, exist_ok=True)

# Load Whisper on GPU
whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Whisper device: {whisper_device}")
whisper_model = whisper.load_model("medium", device=whisper_device)

VOICE_MAP = {
    "ta": "ta-IN-PallaviNeural",
    "hi": "hi-IN-MadhurNeural",
    "ml": "ml-IN-SobhanaNeural",
    "en": "en-US-GuyNeural",
    "fr": "fr-FR-DeniseNeural",
}

LANG_NAME_MAP = {
    "ta": "tamil",
    "hi": "hindi",
    "ml": "malayalam",
    "en": "english",
    "fr": "french",
}

WHISPER_LANG = {
    "ta": "ta", "hi": "hi", "ml": "ml", "en": "en", "fr": "fr",
}

ATEMPO_MAX   = 1.8
NATURAL_ZONE = 1.15


# ─────────────────────────────────────────────────────────────
# TRANSCRIPTION
# ─────────────────────────────────────────────────────────────
def transcribe(audio_path: str, source_lang: str) -> list:
    kwargs = {}
    if source_lang != "auto" and source_lang in WHISPER_LANG:
        kwargs["language"] = WHISPER_LANG[source_lang]
    result = whisper_model.transcribe(audio_path, **kwargs)
    return result["segments"]


# ─────────────────────────────────────────────────────────────
# MERGE SHORT SEGMENTS
# ─────────────────────────────────────────────────────────────
def merge_segments(segments: list, min_words: int = 5, max_window: float = 7.0) -> list:
    merged, i = [], 0
    while i < len(segments):
        seg = {**segments[i]}
        while (
            len(seg["text"].split()) < min_words
            and i + 1 < len(segments)
            and segments[i + 1]["end"] - seg["start"] <= max_window
        ):
            i += 1
            seg["text"] = seg["text"].rstrip() + " " + segments[i]["text"].lstrip()
            seg["end"]  = segments[i]["end"]
        merged.append(seg)
        i += 1
    return merged


# ─────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────
FILLERS = re.compile(
    r'\b(um+|uh+|ah+|er+|hmm+|like|you know|i mean|basically|'
    r'actually|literally|okay so|so yeah|right so|you see)\b',
    re.IGNORECASE
)

def clean(text: str) -> str:
    text = FILLERS.sub("", text).strip()
    text = re.sub(r' {2,}', ' ', text)
    if text and text[-1] not in ".!?,;:":
        text += "."
    return text


# ─────────────────────────────────────────────────────────────
# TRANSLATION
# ─────────────────────────────────────────────────────────────
def translate_one(text: str, target_lang: str, retries: int = 3) -> str:
    lang_name = LANG_NAME_MAP.get(target_lang, "english")
    for attempt in range(retries):
        try:
            result = GoogleTranslator(source="auto", target=lang_name).translate(text)
            return result if result else text
        except Exception as e:
            print(f"   ⚠️  Translation attempt {attempt+1} failed: {e}")
            time.sleep(1.5)
    return text


def translate_segments(segments: list, target_lang: str) -> list:
    BATCH = 3
    results = [""] * len(segments)
    texts   = [clean(s["text"]) for s in segments]
    for i in range(0, len(texts), BATCH):
        batch  = texts[i : i + BATCH]
        joined = " | ".join(batch)
        try:
            translated_joined = translate_one(joined, target_lang)
            parts = [p.strip() for p in translated_joined.split("|")]
            if len(parts) == len(batch):
                for j, p in enumerate(parts):
                    results[i + j] = p
            else:
                for j, t in enumerate(batch):
                    results[i + j] = translate_one(t, target_lang)
        except Exception:
            for j, t in enumerate(batch):
                results[i + j] = translate_one(t, target_lang)
    return results


# ─────────────────────────────────────────────────────────────
# GRAMMAR FIX
# ─────────────────────────────────────────────────────────────
def fix_grammar(text: str, lang: str) -> str:
    if not text or not text.strip():
        return text
    text = text.strip()
    text = re.sub(r' {2,}', ' ', text)

    # Only remove doubled words for Latin script languages
    if lang in ("en", "fr"):
        text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)

    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])(?=[^\s])', r'\1 ', text)
    text = text.replace('|', '').strip()
    text = re.sub(r' {2,}', ' ', text)

    if lang == "hi":
        for pp in ["को", "में", "से", "पर", "ने", "का", "के", "की", "तक", "और"]:
            text = re.sub(rf'{pp}\s+{pp}', pp, text)
        for aux in ["है", "हैं", "था", "थे", "थी", "हो", "हूं", "गया", "गई", "गए"]:
            text = re.sub(rf'{aux}\s+{aux}', aux, text)
        text = re.sub(r'(है|हैं|था|थी|थे)\s+(है|हैं|था|थी|थे)', r'\1', text)
        if text and text[-1] not in "।.!?":
            text += "।"

    elif lang == "ta":
        for particle in ["மற்றும்", "அல்லது", "ஆனால்", "என்று", "இல்", "உடன்"]:
            text = re.sub(rf'{particle}\s+{particle}', particle, text)
        text = re.sub(r'\b(is|are|was|were|the|a|an|and|or|but)\b', '', text)
        text = re.sub(r' {2,}', ' ', text).strip()
        if text and text[-1] not in ".!?":
            text += "."

    elif lang == "ml":
        for particle in ["എന്ന്", "ആണ്", "ഉണ്ട്", "കഴിഞ്ഞ്", "ആയി", "എന്നാൽ"]:
            text = re.sub(rf'{particle}\s+{particle}', particle, text)
        if text and text[-1] not in ".!?":
            text += "."

    return re.sub(r' {2,}', ' ', text).strip()


def post_process_all(translated: list, lang: str) -> list:
    return [fix_grammar(t, lang) for t in translated]


# ─────────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────────
async def generate_tts(text: str, path: str, lang: str):
    voice = VOICE_MAP.get(lang, "en-US-GuyNeural")
    await edge_tts.Communicate(text, voice).save(path)


# ─────────────────────────────────────────────────────────────
# AUDIO HELPERS
# ─────────────────────────────────────────────────────────────
def build_atempo_chain(speed: float) -> str:
    if 0.5 <= speed <= 2.0:
        return f"atempo={speed:.4f}"
    filters, remaining = [], speed
    if speed < 0.5:
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.4f}")
    else:
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


def get_audio_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def stretch_to_window(input_mp3: str, output_wav: str, window_sec: float) -> bool:
    tts_dur = get_audio_duration(input_mp3)
    if tts_dur <= 0:
        return False
    speed = tts_dur / window_sec
    if speed < 0.85:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_mp3,
            "-af", f"apad=whole_dur={window_sec}",
            output_wav,
        ], check=True, capture_output=True)
        return True
    if speed > ATEMPO_MAX:
        atempo = build_atempo_chain(ATEMPO_MAX)
        subprocess.run([
            "ffmpeg", "-y", "-i", input_mp3,
            "-af", f"{atempo},atrim=0:{window_sec:.3f},asetpts=PTS-STARTPTS",
            output_wav,
        ], check=True, capture_output=True)
        return True
    atempo = build_atempo_chain(speed)
    subprocess.run([
        "ffmpeg", "-y", "-i", input_mp3,
        "-af", atempo,
        output_wav,
    ], check=True, capture_output=True)
    return True


def build_final_audio(segment_wavs: list, video_duration: float, output_path: str):
    if not segment_wavs:
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"anullsrc=r=44100:cl=mono:d={video_duration}",
            output_path,
        ], check=True)
        return
    inputs, filter_parts = [], []
    for idx, (start_sec, wav_path) in enumerate(segment_wavs):
        inputs    += ["-i", wav_path]
        delay_ms   = int(start_sec * 1000)
        filter_parts.append(
            f"[{idx}]adelay={delay_ms}|{delay_ms},"
            f"apad=whole_dur={video_duration}[a{idx}]"
        )
    n = len(segment_wavs)
    mix = "".join(f"[a{i}]" for i in range(n))
    filter_parts.append(
        f"{mix}amix=inputs={n}:duration=first:normalize=0[out]"
    )
    subprocess.run(
        ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", ";".join(filter_parts),
            "-map", "[out]", output_path,
        ],
        check=True,
    )


# ─────────────────────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    target_lang: str = "ta",
    source_lang: str = "auto",
):
    try:
        print("\n🎬 === NEW JOB ===")

        # 1. Save
        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(video_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        print(f"📁 Saved: {file.filename}")

        # 2. Extract audio (16kHz mono for Whisper)
        print("🎧 Extracting audio...")
        audio_path = os.path.join(TEMP_FOLDER, "original.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", audio_path],
            check=True, capture_output=True,
        )

        # 3. Transcribe
        print("🧠 Transcribing...")
        raw_segs = transcribe(audio_path, source_lang)
        print(f"   → {len(raw_segs)} raw segments")

        # 4. Merge short segments
        segments = merge_segments(raw_segs)
        print(f"   → {len(segments)} after merge")
        for s in segments:
            print(f"      [{s['start']:.1f}-{s['end']:.1f}s] {s['text'][:55]!r}")

        # 5. Translate
        print(f"🌍 Translating to [{target_lang}]...")
        translated = translate_segments(segments, target_lang)

        # 6. Grammar fix
        print("✍️  Post-processing grammar...")
        polished = post_process_all(translated, target_lang)
        for seg, raw, pol in zip(segments, translated, polished):
            print(f"   [{seg['start']:.1f}s]  {raw[:45]!r}")
            if raw != pol:
                print(f"          → {pol[:45]!r}")

        # 7. TTS per segment (concurrent)
        print("🔊 Generating TTS...")
        tts_jobs = []
        for idx, (seg, txt) in enumerate(zip(segments, polished)):
            if not txt.strip():
                continue
            path = os.path.join(TEMP_FOLDER, f"tts_{idx}.mp3")
            tts_jobs.append((idx, seg["start"], seg["end"], path, txt))
        await asyncio.gather(*[
            generate_tts(txt, path, target_lang)
            for (_, _, _, path, txt) in tts_jobs
        ])

        # 8. Stretch each TTS to window
        print("⏱️  Stretching TTS to segment windows...")
        segment_wavs = []
        for (idx, start, end, tts_path, _) in tts_jobs:
            if not os.path.exists(tts_path):
                continue
            window         = max(0.5, end - start)
            stretched_path = os.path.join(TEMP_FOLDER, f"stretched_{idx}.wav")
            tts_dur        = get_audio_duration(tts_path)
            speed          = round(tts_dur / window, 3) if window > 0 else 1.0
            print(f"   seg {idx}: window={window:.2f}s  tts={tts_dur:.2f}s  speed={speed:.2f}x")
            ok = stretch_to_window(tts_path, stretched_path, window)
            if ok and os.path.exists(stretched_path):
                segment_wavs.append((start, stretched_path))

        # 9. Mix dubbed audio
        print("🎚️  Mixing dubbed audio...")
        video_dur  = get_audio_duration(video_path)
        mixed_path = os.path.join(TEMP_FOLDER, "final_audio.wav")
        build_final_audio(segment_wavs, video_dur, mixed_path)

        # 10. Merge dubbed audio into video (intermediate)
        print("🎬 Merging dubbed audio into video...")
        final_video = os.path.join(OUTPUT_FOLDER, "final_output.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", video_path, "-i", mixed_path,
            "-c:v", "copy",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest", final_video,
        ], check=True)

        # 11. Wav2Lip + Demucs BGM
        print("💋 Running Wav2Lip + BGM restoration...")
        lip_synced_video = os.path.join(OUTPUT_FOLDER, "lip_synced_output.mp4")
        lip_sync_done    = False

        try:
            from wav2lip_runner import run_wav2lip
            success = run_wav2lip(
                video_path          = final_video,
                audio_path          = mixed_path,
                output_path         = lip_synced_video,
                original_video_path = video_path,      # original upload → Demucs BGM
            )
            if success:
                serve_video   = lip_synced_video
                lip_sync_done = True
                print("✅ Lip sync + BGM complete!")
            else:
                serve_video = final_video
                print("⚠️  Lip sync failed — serving dubbed video")
        except Exception as e:
            print(f"⚠️  Wav2Lip error: {e}")
            serve_video = final_video

        print("✅ All done!")
        return {
            "message":  "Success",
            "video":    f"outputs/{os.path.basename(serve_video)}",
            "segments": len(segments),
            "duration": round(video_dur, 1),
            "lip_sync": lip_sync_done,
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}


@app.get("/outputs/{filename}")
def get_video(filename: str):
    return FileResponse(os.path.join(OUTPUT_FOLDER, filename), media_type="video/mp4")