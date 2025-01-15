from transformers import AutoProcessor, BarkModel
import torch

processor = AutoProcessor.from_pretrained("suno/bark")
# processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = model.to(device)

voice_preset = "v2/zh_speaker_5"
inputs = processor("你好，我叫谷纪豪。[laugh]", voice_preset=voice_preset)
inputs = inputs.to(device)
audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

import scipy
sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)