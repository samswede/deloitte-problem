from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transcription import transcribe_audio

# Load the processor and model
processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en") # whisper-tiny.en  # openai/whisper-large-v3
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny.en") # whisper-tiny.en  # openai/whisper-large-v3

# Specify the path to your audio file
audio_path = "audio/EarningsCall.wav"

# Call the function to transcribe the audio
predicted_text = transcribe_audio(audio_path, processor, model)

# Print or use the predicted text
print(predicted_text)
