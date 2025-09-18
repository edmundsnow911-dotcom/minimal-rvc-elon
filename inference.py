import os
import torch
import numpy as np
import librosa
import soundfile as sf

# RVC용 핵심 함수들
def load_hubert_model():
    from hubert_model import HubertSoft  # minimal-rvc-elon 폴더 안에 있다고 가정
    model_path = "hubert/hubert_soft.pt"
    model = HubertSoft(model_path)
    model.eval()
    return model

def convert_voice(audio_path):
    # 경로 설정
    model_name = "ElonMusk_90s"
    model_path = f"models/elon/{model_name}.pth"
    index_path = f"models/elon/added_IVF115_Flat_nprobe_1_{model_name}_v2.index"
    output_path = "/tmp/output.wav"

    # 오디오 로딩
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    if len(audio) == 0:
        raise ValueError("Audio is empty")

    # Torch Tensor로 변환
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0)

    # ✅ 여기에 모델 로딩 및 변조 로직 추가 (추후 개선 지점)
    # 예시 목적: 단순히 원본 오디오를 저장 (RVC 추론은 향후 연결)
    sf.write(output_path, audio, sr)

    return output_path
