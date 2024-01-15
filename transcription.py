from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torchaudio
import torch

def transcribe_audio(audio_path, processor, model):
    # Load the audio file using torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Preprocess the audio using the processor
    inputs = processor(waveform, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(input_values=inputs.input_values)

    # Extract the predicted text from the model's output
    predicted_ids = outputs.logits.argmax(-1)
    predicted_text = processor.batch_decode(predicted_ids)

    return predicted_text
