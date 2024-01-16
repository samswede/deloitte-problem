from transformers import WhisperProcessor, WhisperForConditionalGeneration

def transcribe_audio(chunks, sample_rate):
    # Load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    transcriptions = []
    for chunk in chunks:
        # Process the chunk
        input_features = processor(chunk.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features

        # Generate token ids
        predicted_ids = model.generate(input_features)

        # Decode token ids to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        transcriptions.append(transcription[0])

    return ' '.join(transcriptions)
