import torchaudio
import torch

def transcribe_audio(audio_path, processor, model):
    try:
        # Load the audio file using torchaudio
        waveform, sample_rate = torchaudio.load(    audio_path, 
                                                    normalize=True)

        print(f"waveform shape: {waveform.shape}")
        print(f"Sample rate: {sample_rate}")

        # Preprocess the audio using the processor
        inputs = processor( waveform.squeeze().numpy(), 
                            return_tensors="pt", 
                            sample_rate=sample_rate)

        # Perform inference
        with torch.no_grad():
           outputs = model(input_values=inputs.input_values)
        
        # Extract the predicted text from the model's output
        predicted_ids = outputs.logits.argmax(-1)
        predicted_text = processor.batch_decode(predicted_ids)

        return predicted_text
    
    except Exception as e:
        print(f"An error occurred: {e}")

        return None

