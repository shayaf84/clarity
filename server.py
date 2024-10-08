import asyncio
import json
import os
import uuid
import logging
import av.container
import threading
import queue
import io

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt

import numpy as np
import cv2
import torch
import torchaudio
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoProcessor, WhisperForConditionalGeneration, HubertForSequenceClassification, ViTImageProcessor, ViTForImageClassification, ViTFeatureExtractor
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from PIL import Image
import av
from av.audio.frame import AudioFrame
from av.video.frame import VideoFrame

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

#
# LOGIC FOR VIDEO PROCESSING: FACE RECOGNITION, EXTRACTION, AND FATIGUE PREDICTION
#

class VideoTransformTrack(MediaStreamTrack):
    """
    Take in a MediaStream and process it frame by frame
    """

    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.model_path = 'xacer/vit-base-patch16-224-fatigue'
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        self.processor = ViTImageProcessor.from_pretrained(self.model_path)
        self.model = ViTForImageClassification.from_pretrained(self.model_path,num_labels=2,ignore_mismatched_sizes=True).cuda()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_path) 
        self.image_mean, self.image_std = self.processor.image_mean, self.processor.image_std

        self.transform = Compose([

            Resize((224,224)),
            ToTensor(),
            Normalize(mean=self.image_mean,std=self.image_std)

        ])
        self.class_value = 1
        self.frame_count = 0

    def crop_bounding_box(self, img, x, y, w, h):
        return img[y:y+h,x:x+w]

    async def recv(self):
        # Retrieve the next input frame
        video_frame: VideoFrame = await self.track.recv()
        outgoing_image: np.ndarray = video_frame.to_ndarray(format="bgr24")
    
        # TODO: Detect if a face is present in the image
        gray = cv2.cvtColor(outgoing_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1,4)
        # TODO: Crop and reshape the bounding box to 224 x 224
        for (x,y,w,h) in faces:
            cv2.rectangle(outgoing_image,(x,y),(x+w,y+h),(255,0,0),2)
            
            # TODO: Pass face image into vision transformer model
            if self.frame_count % 12 == 0:
                cropped_image = self.crop_bounding_box(outgoing_image,x,y,w,h)
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
                cropped_pil = Image.fromarray(cropped_image)
                transformed_pil = self.transform(cropped_pil)
                input_tensor = transformed_pil.unsqueeze(0)
                with torch.no_grad():
                    input_tensor = input_tensor.cuda()
                    output_logits = self.model(input_tensor).logits.cpu()
                predicted_class = torch.argmax(output_logits).item()
                if predicted_class == 1:
                    self.class_value = "Active"
                elif predicted_class == 0:
                    self.class_value = "Fatigued"
            
            # TODO: Compute saliency map for face

            # TODO: Draw bounding box and saliency map onto image
            cv2.putText(outgoing_image, f'{self.class_value}',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            

        self.frame_count += 1

        # Reconstruct and return the new frame
        new_frame = VideoFrame.from_ndarray(outgoing_image, format="bgr24")
        new_frame.pts = video_frame.pts
        new_frame.time_base = video_frame.time_base

        return new_frame

#
# LOGIC FOR AUDIO RECORDING
#

class CustomMediaRecorder:
    instance: "CustomMediaRecorder" = None

    def __init__(self):
        self.counter = 0
        self.reset_container()
        self.track = None
        self.task = None
        self.recording = False

        CustomMediaRecorder.instance = self

    def add_track(self, track: MediaStreamTrack):
        self.track = track
    
    def reset_container(self):
        self.filename = f"recording_{self.counter:03}.mp3"
        self.container = av.open(self.filename, mode="w")
        self.stream = self.container.add_stream("mp3")
        self.counter += 1
    
    def start_recording(self):
        self.recording = True

    def stop_recording(self):
        self.recording = False
        self.container.close()
        old_filename = self.filename
        self.reset_container()
        return old_filename

    async def start(self):
        self.task = asyncio.ensure_future(self.run_track())
    
    async def stop(self):
        if self.task is not None:
            self.task.cancel()
            self.task = None

            for packet in self.stream.encode(None):
                self.container.mux(packet)
        
        self.container.close()

        self.track = None
        self.stream = None
        self.container = None
    
    async def run_track(self):
        while True:
            frame: AudioFrame = await self.track.recv()
            if self.recording:
                for packet in self.stream.encode(frame):
                    self.container.mux(packet)

#
# LOGIC FOR AUDIO PROCESSING
#

class AudioAnalyzer:
    instance: "AudioAnalyzer"

    def __init__(self):
        # TODO: load Hugging Face models
        self.hubert = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-er").cuda()
        self.to_spectrogram = torchaudio.transforms.Spectrogram(n_fft=1024,return_complex=True, power=None, hop_length=256).cuda()
        self.from_spectrogram = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256).cuda()
        self.mel_scale = torchaudio.transforms.MelScale(n_mels=128, n_stft=513).cuda()

        self.noise_rescale = 1.0 / torch.linspace(1.0, 16.0, self.to_spectrogram.n_fft // 2 + 1).cuda()
        self.model_path = "openai/whisper-tiny.en"
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path).cuda()

        from transformers import pipeline
        self.faster_whisper = pipeline("automatic-speech-recognition", "openai/whisper-tiny", torch_dtype=torch.float16, device="cuda:0")

        AudioAnalyzer.instance = self

    def run_audio_analysis(self, filename: str, output_queue: queue.Queue):
        # load audio file

        waveform, samplerate = torchaudio.load(filename)
        resampler = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=16000)
        waveform = waveform[0:1]
        waveform: torch.Tensor = resampler.forward(waveform).cuda()

        all_text: str = self.faster_whisper(filename)["text"]        
        all_text = all_text.strip()

        # obtain initial diagnosis
        with torch.no_grad():
            logits: torch.Tensor = self.hubert(waveform).logits
        
        # convert to spectrogram
        spectrogram = self.to_spectrogram(waveform)

        print("Computed spectrogram:", spectrogram.shape)
        print("Noise rescaler:", self.noise_rescale.shape)

        # compute saliency map
        gradient_list = []
        for i in range(10):
            noisy_data = torch.randn_like(spectrogram) * 0.1 * self.noise_rescale.unsqueeze(0).unsqueeze(-1).repeat(spectrogram.shape[0], 1, spectrogram.shape[-1])
            noisy_data = spectrogram + noisy_data
            noisy_data.requires_grad = True
            noisy_waveform = self.from_spectrogram(noisy_data)
            noisy_label = self.hubert(noisy_waveform).logits[..., 3]
            self.hubert.zero_grad()
            noisy_label.backward()
            noisy_gradient = noisy_data.grad.clone().squeeze() * self.noise_rescale.unsqueeze(-1).repeat(1, spectrogram.shape[-1])
            gradient_list += [noisy_gradient]
        
        saliency_map = torch.mean(torch.stack(gradient_list), dim=0).detach().abs()
        saliency_map = self.mel_scale.forward(saliency_map)
        saliency_map = torch.flip(saliency_map, (1,))

        blur_fn = torchvision.transforms.GaussianBlur(15, sigma=(0.1, 2)).cuda()
        saliency_map = blur_fn(blur_fn(saliency_map.unsqueeze(0))).squeeze().cpu()

        # prepare everything for client

        # rescale
        power_spectrogram = torch.real(spectrogram).pow(2) + torch.imag(spectrogram).pow(2)
        power_spectrogram = power_spectrogram.squeeze()

        mel_spectrogram = self.mel_scale.forward(power_spectrogram)
        mel_spectrogram = mel_spectrogram.clip(min=1e-5)
        mel_spectrogram = mel_spectrogram.log10()
        mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())
        mel_spectrogram = mel_spectrogram * 255
        mel_spectrogram = mel_spectrogram.cpu().numpy().astype(np.uint8)

        # mel_spectrogram = cv2.applyColorMap(mel_spectrogram, cv2.COLORMAP_MAGMA)
        # mel_spectrogram = cv2.cvtColor(mel_spectrogram, cv2.COLOR_BGR2RGBA)

        plt.style.use("dark_background")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(mel_spectrogram, cmap="inferno", origin="lower")

        cmap2 = matplotlib.cm.get_cmap("viridis")
        cmap2._init()
        alphas = np.linspace(0, 1.0, cmap2.N+3)
        cmap2._lut[:,-1] = alphas

        ax.imshow(
            saliency_map, 
            cmap=cmap2, 
            interpolation="bilinear",
            origin="lower"
        )
        ax.set_xlabel("Time (s)")
        locs, labels = plt.xticks()
        labels = [round(float(item)*256/16000,1) for item in locs]        
        plt.xticks(locs, labels)
        plt.xlim([0, mel_spectrogram.shape[-1]])
        ax.set_ylabel("Frequency (log)")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches='tight')
        buf.seek(0)
        data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        data = cv2.imdecode(data, 1)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGBA)

        output_queue.put({
            "waveform": waveform.tolist(),
            "spectrogramImageData": data.flatten().tolist(),
            "spectrogramHeight": data.shape[0],
            "spectrogramWidth": data.shape[1],
            "saliency": saliency_map.tolist(),
            "logits": logits.tolist(),
            "labels": ["Neutral", "Happy", "Angry", "Sad"],
            "transcription": all_text
        })

    async def run_audio_analysis_threaded(self, filename: str):
        out_queue = queue.Queue()
        task = threading.Thread(target=self.run_audio_analysis, args=(filename, out_queue))
        task.start()

        while task.is_alive():
            await asyncio.sleep(0.1)
        
        return out_queue.get()

#
# LOGIC FOR WEBRTC VIDEO AND AUDIO STREAMING
#

async def get_index_html(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def get_main_js(request):
    content = open(os.path.join(ROOT, "main.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def get_style_css(request):
    content = open(os.path.join(ROOT, "style.css"), "r").read()
    return web.Response(content_type="text/css", text=content)

async def post_start_recording(request):
    CustomMediaRecorder.instance.start_recording()
    return web.Response(content_type="application/json", text="{\"success\": \"true\"}")

async def get_stop_recording(request):
    saved_filename = CustomMediaRecorder.instance.stop_recording()
    analysis = await AudioAnalyzer.instance.run_audio_analysis_threaded(saved_filename)
    return web.Response(content_type="application/json", text=json.dumps(analysis))

async def post_offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)
    
    custom_recorder = CustomMediaRecorder()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "audio":
            custom_recorder.add_track(track)
        
        elif track.kind == "video":
            pc.addTrack(VideoTransformTrack(relay.subscribe(track)))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await custom_recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await custom_recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp, 
            "type": pc.localDescription.type
        }),
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

def main():
    logging.basicConfig(level=logging.INFO)

    # Initialize the audio analysis module
    _audio_analyzer = AudioAnalyzer()
    
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", get_index_html)
    app.router.add_get("/main.js", get_main_js)
    app.router.add_get("/style.css", get_style_css)
    app.router.add_post("/start", post_start_recording)
    app.router.add_get("/stop", get_stop_recording)
    app.router.add_post("/offer", post_offer)
    web.run_app(app, host="127.0.0.1", port="8080")

if __name__ == "__main__":
    main()