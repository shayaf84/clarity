# Clarity
Hack GT '24 - An early warning system that detects quantifiable physiological cues (facial recognition, speech) for fatigue in doctors

Link to Devpost: https://devpost.com/software/clarity-kuv9xn

Model Name on HuggingFace: https://huggingface.co/xacer/vit-base-patch16-224-fatigue

## Inspiration
We all know the cases of the many lawsuits that come out of misdiagnosis errors. For example, recently here in Atlanta, Emory Healthcare was ordered to pay $38 M for forgetting to run a crucial heart CT scan that would have identified a complication with the patient’s heart transplant. I (Shaya) was in a Psychology 1101 lecture on Friday morning and the subject of discussion was “involuntary automaticity” → where an individual who is used to performing a task frequently develops a “second nature habit”, and allocates less of their cognitive capacity on that given task. After discussing with the rest of the team, we were curious about its implications in misdiagnosis errors, and through further research, we realized that fatigue can worsen this decreased cognitive capacity, as when one is tired, they are more susceptible to fall into this automatic process of thinking, and in a way that is prone to errors. Furthermore, we realized that prolonged fatigue can also lead to burnout, as well as, depression in doctors and that according to the Stanford School of Medicine, doctors have 3-5 times the suicide rate of the general public. Hence, we realized that these errors are not intentional, and are a result of systems not being in place to support fatigued doctors. We aimed to change that by examining how we can work to reduce the burden placed on these doctors.

## What it does
Clarity is a multi-modal suite aimed at detecting fatigue in doctors. It captures your facial expression in real-time and returns a bounding box around it indicating whether you are fatigued or active. In addition, it also captures an audio recording, and returns a classification of fatigue/active, a mel-spectrogram of the recording, and a saliency map which highlights the key features of the spectrogram that contributed to the prediction (a layer of explainability).

## How we built it
Clarity uses a two-track pipeline: one for the video data and one for the audio data.

 - **Video Track**: A native WebRTC library is used to stream real time camera data to the backend. Individual frames are piped into a pre-trained Haar Cascade Classifier to detect and localize the face with a bounding box. From there, the image is cropped to only include the face likeness and is classified as active or fatigued by a visual transformer that we trained on the Facial Expression of Fatigues (FEF) dataset on Kaggle. To improve generalization, we leveraged a pre-trained self-supervised model on ImageNet, manually enabled a subset of network layers, and fine tuned on the facial expression dataset. The prediction is overlaid onto the bounding box with OpenCV and streamed the result back to the client with WebRTC.
 - **Audio Track**: The WebRTC stream is extended to include audio, which is conditionally recorded into an MP3 file based on user input. When the user stops recording, the audio is loaded as a waveform, then transformed into a Mel spectrogram using the Short Time Fourier Transform (STFT). The speech representation is fed into a pre-trained HuBERT-base neural network, which computes various emotion labels. A saliency map is computed via the smoothed gradient of the spectrogram with respect to the model output. The spectrogram is overlaid with the saliency map, and the combined plot and diagnosis are returned to the client for display. We separately compute a transcription with a pretrained whisper-tiny model to help the user correlate parts of the spectrogram with the spoken words.

## Challenges we ran into
When training the facial expression classifier model, we ran into the problem that the distribution of our training dataset was too small compared to the distribution of the actual data we needed to evaluate on. To improve generalization, we transitioned to start with a baseline vision transformer model trained on ImageNet, then through trial-and-error tuned the learning rate, number of active layers, and number of training epochs until our train accuracy matched our test accuracy.
Originally, we planned on using a local Macbook Pro to run the server and neural network inference. However, Mac has a specialized hardware acceleration platform called Metal, which is incompatible with our neural network frameworks. Without another powerful local device, we ended up remotely connecting to a machine in Jacksonville, FL (hometown of a team member) containing an NVIDIA 4090 RTX. We were surprised that the latency of streaming video to and from Jacksonville was low enough to create a viable proof of concept.
We had problems making scientific plots of our spectrogram with JavaScript, which is why we eventually transitioned to server-side rendering of plots and overlays with Matplotlib.

## Accomplishments that we're proud of
 - Building a multi-modal suite consisting of several models that detect in real time
 - Fine-tuning a visual transformer on fatigue detection
 - Generating saliency maps to facilitate explainable AI.

## What we learned
 - Multi-threading in Python to reduce server freezing during compute-heavy audio speech processing
 - Real-time audio/video streaming from client to server with WebRTC
 - Manually fine-tuning self-supervised models by enabling only specific layers
 - Speculative decoding for automatic speech recognition
 - Computing saliency maps with the smoothed gradient method
 - Existing research on psychological basis for involuntary automaticity, fatigue, and preventable errors

## What's next for Clarity
There are several next steps:

 - **Saliency Maps for Video**: Add saliency maps overlaid on top of the video feed to identify regions that contributed most to the prediction of fatigued or active.
 - **Email Notifications**: When flags are detected for a sustained period of time, automatically notify another staff member through via email to mitigate the risk of an error being formed due to high fatigue (eg: have the doctor assisted by another doctor)
 - **Integrate Doctor’s Notes**: Use the doctor’s notes in the EHR as another input source and run sentiment analysis to identify possible fatigue.
 - **Scalability**: Work to integrate this system in existing EHR solutions such that a doctor can use this platform as they are entering patient information into the EHR (prior to making a diagnosis)