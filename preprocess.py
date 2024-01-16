import torchaudio
import torch

from pydub import AudioSegment

def preprocess_audio(audio_path, chunk_length_s=30, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert stereo to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample the waveform to target_sample_rate if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    # Chunking the audio
    num_samples_per_chunk = sample_rate * chunk_length_s
    total_samples = waveform.shape[1]
    chunks = [waveform[:, i:i+num_samples_per_chunk] for i in range(0, total_samples, num_samples_per_chunk)]

    return chunks, sample_rate



def convert_m4a_to_wav(input_path, output_path):
    # Load the m4a file
    audio = AudioSegment.from_file(input_path, format="m4a")

    # Export as wav
    audio.export(output_path, format="wav")

    return output_path
