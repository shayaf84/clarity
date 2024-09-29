import torch
import torchaudio
from transformers import AutoProcessor, WhisperForConditionalGeneration
# from datasets import load_dataset
from datasets import Dataset

def run_audio_analysis(filename: str):
    # load audio file
    waveform, samplerate = torchaudio.load(filename)
    resampler = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=16000)
    waveform = waveform[0:1]
    waveform: torch.Tensor = resampler.forward(waveform) #.cuda()

    waveform = waveform.tolist()
    data_dict = {
        'audio': waveform,
    }

    ds = Dataset.from_dict(data_dict)

    transcribe(ds)


def transcribe(ds):
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    # ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    # print(ds[0]["audio"])

    inputs = processor(ds[0]["audio"], return_tensors="pt")
    input_features = inputs.input_features

    generated_ids = model.generate(inputs=input_features)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print(transcription)


if __name__ == "__main__":

    run_audio_analysis("audio_file.mp3")



