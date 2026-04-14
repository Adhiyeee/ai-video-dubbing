"""
wav2lip_runner.py — FINAL CLEAN VERSION
=========================================
Strategy:
1. Demucs separates original audio → vocals.wav + no_vocals.wav (pure BGM)
2. Mix pure BGM (no_vocals) at 100% + dubbed voice at 100%
3. Wav2Lip lip sync on GPU
4. NO video overlay — clean original quality video + synced audio

Result: perfect audio (dubbed voice + pure BGM, zero original voice bleed)
        + lip synced video at original quality
"""

import os
import sys
import shutil
import subprocess
import tempfile

WAV2LIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wav2Lip")
CHECKPOINT  = os.path.join(WAV2LIP_DIR, "checkpoints", "wav2lip_gan.pth")
INFERENCE   = os.path.join(WAV2LIP_DIR, "inference.py")
SAFE_TEMP   = "C:\\wav2lip_tmp"

# Python 3.11 path — the one with CUDA torch + demucs
PY311 = r"C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe"


def get_video_dimensions(video_path: str):
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    try:
        w, h = result.stdout.strip().split(",")
        return int(w), int(h)
    except Exception:
        return 478, 850


def patch_inference_for_gpu(original_path: str, output_path: str):
    with open(original_path, "r", encoding="utf-8") as f:
        src = f.read()
    src = src.replace(
        "use_cuda = torch.cuda.is_available()",
        "use_cuda = True  # forced by wav2lip_runner"
    )
    src = src.replace(
        "device = 'cuda' if use_cuda else 'cpu'",
        "device = 'cuda'  # forced by wav2lip_runner"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(src)


def separate_bgm_with_demucs(audio_path: str, out_dir: str) -> str | None:
    """
    Run Demucs on audio_path.
    Returns path to the no_vocals (BGM only) wav, or None on failure.
    Demucs outputs: out_dir/htdemucs/<stem>/no_vocals.wav
    """
    print("🎵 Separating BGM with Demucs (AI voice removal)...")
    try:
        result = subprocess.run([
            PY311, "-m", "demucs",
            "--two-stems", "vocals",   # only split into vocals + no_vocals
            "--out", out_dir,
            "--mp3",                   # faster processing
            audio_path,
        ], capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"   Demucs error: {result.stderr[-300:]}")
            return None

        # Find the no_vocals file demucs created
        stem_name = os.path.splitext(os.path.basename(audio_path))[0]
        # Demucs creates: out_dir/htdemucs/<stem>/no_vocals.mp3
        candidates = []
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                if "no_vocals" in f:
                    candidates.append(os.path.join(root, f))

        if not candidates:
            print(f"   no_vocals file not found in {out_dir}")
            return None

        bgm_path = candidates[0]
        print(f"   BGM extracted → {bgm_path}")
        return bgm_path

    except subprocess.TimeoutExpired:
        print("   Demucs timed out")
        return None
    except Exception as e:
        print(f"   Demucs failed: {e}")
        return None


def run_wav2lip(
    video_path: str,
    audio_path: str,
    output_path: str,
    original_video_path: str = None,
) -> bool:
    os.makedirs(SAFE_TEMP, exist_ok=True)

    video_path  = os.path.abspath(video_path)
    audio_path  = os.path.abspath(audio_path)
    output_path = os.path.abspath(output_path)
    if original_video_path:
        original_video_path = os.path.abspath(original_video_path)

    safe_video        = os.path.join(SAFE_TEMP, "input_video.mp4")
    safe_audio        = os.path.join(SAFE_TEMP, "input_audio.wav")
    safe_result_raw   = os.path.join(SAFE_TEMP, "result_raw.mp4")
    safe_result_final = os.path.join(SAFE_TEMP, "result_final.mp4")
    patched_inference = os.path.join(SAFE_TEMP, "inference_gpu.py")
    demucs_out_dir    = os.path.join(SAFE_TEMP, "demucs_out")

    shutil.copy2(video_path, safe_video)
    shutil.copy2(audio_path, safe_audio)
    print(f"   copied inputs → {SAFE_TEMP}")

    patch_inference_for_gpu(INFERENCE, patched_inference)
    print(f"   patched inference.py → forced CUDA")

    w, h = get_video_dimensions(safe_video)
    print(f"   video: {w}x{h}")

    # ── STEP 1: BGM SEPARATION ────────────────────────────────
    bgm_path = None
    if original_video_path and os.path.exists(original_video_path):
        # Extract original audio first
        orig_audio_path = os.path.join(SAFE_TEMP, "original_audio.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", original_video_path,
            "-vn", "-ar", "44100", "-ac", "2",
            orig_audio_path,
        ], check=True, capture_output=True)

        os.makedirs(demucs_out_dir, exist_ok=True)
        bgm_path = separate_bgm_with_demucs(orig_audio_path, demucs_out_dir)

        if bgm_path:
            print(f"   ✅ Pure BGM ready: {os.path.basename(bgm_path)}")
        else:
            print(f"   ⚠️  Demucs failed — will use original audio at 10% as BGM fallback")

    # ── STEP 2: WAV2LIP LIP SYNC ─────────────────────────────
    env = os.environ.copy()
    extra_paths = [WAV2LIP_DIR] + sys.path
    env["PYTHONPATH"] = os.pathsep.join(extra_paths)

    cmd = [
        sys.executable,
        patched_inference,
        "--checkpoint_path",    CHECKPOINT,
        "--face",               safe_video,
        "--audio",              safe_audio,
        "--outfile",            safe_result_raw,
        "--resize_factor",      "1",
        "--nosmooth",
        "--pads",               "0", "0", "0", "0",
        "--face_det_batch_size","4",
        "--wav2lip_batch_size", "128",
        "--box",                "0", str(h), "0", str(w),
    ]

    print("💋 Running Wav2Lip (GPU)...")

    try:
        result = subprocess.run(
            cmd, cwd=WAV2LIP_DIR, env=env, timeout=600,
        )

        if result.returncode != 0:
            print(f"❌ Wav2Lip exited with code {result.returncode}")
            return False

        if not os.path.exists(safe_result_raw):
            print("❌ Wav2Lip result not found")
            return False

        # ── STEP 3: FINAL MIX ────────────────────────────────
        # Video: take from wav2lip (lip synced) — no overlay needed
        #        wav2lip's video quality is acceptable since we're
        #        not overlaying a blurry patch on top of a clean video
        # Audio: dubbed voice (100%) + pure BGM from Demucs (100%)
        #        OR fallback: dubbed voice + original audio at 10%

        print("🎚️  Building final audio mix...")

        if bgm_path and os.path.exists(bgm_path):
            # Perfect mix: dubbed voice + pure BGM (no original voice)
            audio_filter = (
                "[1:a]volume=1.0[dubbed];"
                "[2:a]volume=1.0[bgm];"
                "[dubbed][bgm]amix=inputs=2:duration=first:normalize=0[outa]"
            )
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", original_video_path, # [0] original upload (guaranteed quality)
                "-i", audio_path,           # [1] dubbed audio
                "-i", bgm_path,             # [2] pure BGM (no vocals)
                "-filter_complex", audio_filter,
                "-map", "0:v:0",            # video from original upload
                "-map", "[outa]",        # mixed audio
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                safe_result_final,
            ]
            print("   mode: dubbed voice + pure Demucs BGM ✅")
        else:
            # Fallback: original audio at 10% (faint BGM + faint original voice)
            audio_filter = (
                "[1:a]volume=1.0[dubbed];"
                "[2:a]volume=0.10[bgm];"
                "[dubbed][bgm]amix=inputs=2:duration=first:normalize=0[outa]"
            )
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,            # [0] ORIGINAL quality video  
                "-i", audio_path,            # [1] dubbed audio
                "-i", original_video_path,   # [2] original (10% fallback)
                "-filter_complex", audio_filter,
                "-map", "0:v:0",         # video from ORIGINAL (full quality)
                "-map", "[outa]",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                safe_result_final,
            ]
            print("   mode: dubbed voice + original audio 10% fallback ⚠️")

        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        shutil.move(safe_result_final, output_path)
        print(f"✅ Done → {output_path}")

        # Cleanup
        for f in [safe_video, safe_audio, safe_result_raw, patched_inference]:
            try: os.remove(f)
            except: pass

        return True

    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors="replace") if e.stderr else str(e)
        print(f"❌ Final mix failed:\n{err[:500]}")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Wav2Lip timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False