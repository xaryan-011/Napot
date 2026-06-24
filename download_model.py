"""
NapNot - Model Downloader & Alert Sound Generator
===================================================
Downloads the dlib shape_predictor_68_face_landmarks model
and generates the alert WAV file used by the detector.

Usage:
    python download_model.py
"""

import os
import sys
import bz2
import struct
import wave
import math
import ssl
import urllib.request

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
MODEL_BZ2 = "shape_predictor_68_face_landmarks.dat.bz2"
MODEL_DAT = "shape_predictor_68_face_landmarks.dat"

ALERT_WAV = "alert.wav"
ALERT_FREQ = 1000       # Hz  (tone frequency)
ALERT_DURATION = 2.0    # seconds
ALERT_SAMPLE_RATE = 44100
ALERT_AMPLITUDE = 30000  # max 32767 for 16-bit


# ─── Alert Sound Generator ──────────────────────────────────────────────────

def generate_alert_sound():
    """Generate a pulsing alarm WAV file (no external dependencies needed)."""
    if os.path.exists(ALERT_WAV):
        print(f"[OK] Alert sound already exists: {ALERT_WAV}")
        return

    print(f"[*] Generating alert sound: {ALERT_WAV} ...")

    n_samples = int(ALERT_SAMPLE_RATE * ALERT_DURATION)
    samples = []

    for i in range(n_samples):
        t = i / ALERT_SAMPLE_RATE
        # Pulsing effect: modulate amplitude with a 4 Hz envelope
        envelope = 0.5 * (1.0 + math.sin(2.0 * math.pi * 4.0 * t))
        # Main tone
        tone = math.sin(2.0 * math.pi * ALERT_FREQ * t)
        # Secondary harmonic for urgency
        harmonic = 0.3 * math.sin(2.0 * math.pi * (ALERT_FREQ * 1.5) * t)
        sample = int(ALERT_AMPLITUDE * envelope * (tone + harmonic))
        # Clamp to 16-bit range
        sample = max(-32767, min(32767, sample))
        samples.append(sample)

    # Write WAV file
    with wave.open(ALERT_WAV, "w") as wav:
        wav.setnchannels(1)           # Mono
        wav.setsampwidth(2)           # 16-bit
        wav.setframerate(ALERT_SAMPLE_RATE)
        for s in samples:
            wav.writeframes(struct.pack("<h", s))

    print(f"[OK] Alert sound generated: {ALERT_WAV}")


# ─── Model Downloader ───────────────────────────────────────────────────────

def download_model():
    """Download and extract the dlib facial landmarks model."""
    if os.path.exists(MODEL_DAT):
        print(f"[OK] Model already exists: {MODEL_DAT}")
        return

    # Download
    if not os.path.exists(MODEL_BZ2):
        print(f"[*] Downloading model from dlib.net ...")
        print(f"    URL: {MODEL_URL}")
        print(f"    This may take a few minutes (~100 MB) ...")

        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                bar_len = 40
                filled = int(bar_len * percent / 100)
                bar = "#" * filled + "." * (bar_len - filled)
                sys.stdout.write(f"\r    [{bar}] {percent:5.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
                sys.stdout.flush()

        # Use unverified SSL context as fallback for systems with cert issues
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl_ctx)
        )
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(MODEL_URL, MODEL_BZ2, reporthook=progress_hook)
        print()  # newline after progress bar
        print(f"[OK] Download complete: {MODEL_BZ2}")
    else:
        print(f"[OK] Compressed model already downloaded: {MODEL_BZ2}")

    # Extract
    print(f"[*] Extracting model (this may take a moment) ...")
    with open(MODEL_BZ2, "rb") as f_in:
        decompressed = bz2.decompress(f_in.read())
    with open(MODEL_DAT, "wb") as f_out:
        f_out.write(decompressed)

    print(f"[OK] Model extracted: {MODEL_DAT}")

    # Clean up compressed file
    os.remove(MODEL_BZ2)
    print(f"[OK] Cleaned up: {MODEL_BZ2}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  NapNot - Setup & Model Downloader")
    print("=" * 60)
    print()

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    generate_alert_sound()
    print()
    download_model()

    print()
    print("=" * 60)
    print("  Setup complete! Run the detector with:")
    print("    python napnot.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
