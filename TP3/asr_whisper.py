import os
import json
import time
import torch
import torchaudio
import whisper  # Import direct de la bibliothèque OpenAI

def load_wav_mono_16k(path: str):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav.squeeze(0), sr

def main():
    audio_path = "TP3/data/call_01.wav"
    vad_path = "TP3/outputs/vad_segments_call_01.json"
    out_path = "TP3/outputs/asr_call_01.json"
    os.makedirs("TP3/outputs", exist_ok=True)

    # 1. Chargement audio
    wav, sr = load_wav_mono_16k(audio_path)
    audio_duration_s = wav.numel() / sr

    # 2. Chargement des segments VAD
    with open(vad_path, "r", encoding="utf-8") as f:
        vad_payload = json.load(f)
    segments = vad_payload["segments"]

    # 3. Chargement du modèle Whisper (on utilise 'base' pour la vitesse)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model ('base') on {device}...")
    model = whisper.load_model("base", device=device)

    t0 = time.time()
    results = []

    print(f"Transcribing {len(segments)} segments...")

    for i, seg in enumerate(segments):
        start_s = float(seg["start_s"])
        end_s = float(seg["end_s"])

        # Découpage du segment
        start_idx = int(start_s * sr)
        end_idx = int(end_s * sr)
        seg_wav = wav[start_idx:end_idx]

        # Transcription du segment (Whisper attend du float32)
        # On désactive fp16 si on est sur CPU pour éviter les warnings
        res = model.transcribe(seg_wav.numpy(), language="en", fp16=(device=="cuda"))
        text = res["text"].strip()

        results.append({
            "segment_id": i,
            "start_s": start_s,
            "end_s": end_s,
            "text": text
        })
        print(f"[{i+1}/{len(segments)}] {text}")

    t1 = time.time()
    elapsed_s = t1 - t0
    rtf = elapsed_s / max(audio_duration_s, 1e-9)

    full_text = " ".join([r["text"] for r in results]).strip()

    # Sauvegarde
    payload = {
        "audio_path": audio_path,
        "model_id": "openai/whisper-base",
        "device": device,
        "audio_duration_s": audio_duration_s,
        "elapsed_s": elapsed_s,
        "rtf": rtf,
        "segments": results,
        "full_text": full_text
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Stats ===")
    print(f"Elapsed: {elapsed_s:.2f}s | RTF: {rtf:.3f}")
    print(f"Transcription: {full_text[:100]}...")

if __name__ == "__main__":
    main()