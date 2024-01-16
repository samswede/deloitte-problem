from preprocess import preprocess_audio, convert_m4a_to_wav
from transcription import transcribe_audio


def main():
    file_name = "PureGym"

    #raw_audio_path = f"raw_audio/{file_name}.m4a"  # Path to your audio file

    # Convert m4a to wav
    #audio_path = convert_m4a_to_wav(raw_audio_path, f"audio_wav/{file_name}.wav")

    audio_path = f"audio_wav/{file_name}.wav"

    # Preprocess the audio and get chunks
    chunks, sample_rate = preprocess_audio(audio_path)

    # Transcribe each chunk and concatenate the results
    transcription = transcribe_audio(chunks, sample_rate)

    print("Transcription:", transcription)

if __name__ == "__main__":
    main()
