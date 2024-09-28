import torch
import torchaudio

from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

from matplotlib import pyplot as plt

def run_analysis(filename: str):
    pass

def main():
    
    model_id = "superb/hubert-base-superb-er"

    model = HubertForSequenceClassification.from_pretrained(model_id).cuda()
    #feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)

    filename = "test_recording.mp3"
    waveform, samplerate = torchaudio.load(filename)
    resample = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=16000)
    waveform = waveform[0:1]
    waveform = resample(waveform).cuda()

    to_spectrogram = torchaudio.transforms.Spectrogram(return_complex=True, power=None).cuda()
    from_spectrogram = torchaudio.transforms.InverseSpectrogram().cuda()

    spectrogram = to_spectrogram(waveform)
    print("spectrogram:", spectrogram.shape)

    gradient_list = []
    for i in range(20):
        noisy_data = torch.randn_like(spectrogram) * 0.001
        noisy_data = spectrogram + noisy_data
        noisy_data.requires_grad = True
        noisy_waveform = from_spectrogram(noisy_data)
        noisy_label = model(noisy_waveform).logits.max()
        model.zero_grad()
        noisy_label.backward()
        noisy_gradient = noisy_data.grad.clone().squeeze()
        gradient_list += [noisy_gradient]

        print("noisy gradient:", noisy_gradient.shape)

        print("resampled waveform", waveform.shape)

    # should be (H,W)
    mean_gradients = torch.mean(torch.stack(gradient_list), dim=0).detach().abs().cpu()
    
    print("mean gradients:", mean_gradients.shape, mean_gradients.dtype)

    plt.imshow(mean_gradients, cmap="viridis",origin="lower")
    plt.show()

    # smoothed gradient technique:
    # 1. compute spectrogram of audio
    # repeat some number of times:
    # 2. add noise
    # 3. inverse spectrogram
    # 4. run through model
    # 5. compute derivative w.r.t. noise
    # compile gradients
    # compute mean and abs
    # return image

    #features = feature_extractor(waveform, sampling_rate=16000, padding=True, return_tensors="pt")
    #model_output = model(**features)

    #print(features["input_values"].shape)
    #print(model_output.logits)
    #pass

if __name__ == "__main__":
    main()