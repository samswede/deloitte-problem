import torchaudio
import torch

def transcribe_audio(audio_path, processor, model):
    try:
        # Load the audio file using torchaudio
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        print("Audio file loaded successfully.")

        # Conversion and Resampling Code...

        # Preprocess the audio using the processor
        try:
            inputs = processor(waveform.squeeze().numpy(), return_tensors="pt", sampling_rate=16000)
            print("Audio preprocessing successful.")
        except Exception as e:
            print(f"Error during audio preprocessing: {e}")
            return None

        # Perform inference
        try:
            with torch.no_grad():
                outputs = model(input_values=inputs.input_values)
            print("Inference successful.")
        except Exception as e:
            print(f"Error during model inference: {e}")
            return None

        # Process model outputs
        try:
            if hasattr(outputs, 'logits'):
                predicted_ids = outputs.logits.argmax(-1)
                print("Logits extraction successful.")
            else:
                print("The model's output has no 'logits' attribute.")
                return None

            if hasattr(processor, 'batch_decode'):
                predicted_text = processor.batch_decode(predicted_ids)
                print("Decoding successful.")
            else:
                print("The processor has no 'batch_decode' method.")
                return None
        except Exception as e:
            print(f"Error during output processing: {e}")
            return None

        return predicted_text
    
    except Exception as e:
        print(f"An overarching error occurred: {e}")
        return None
