import asyncio
import json
import os
import uuid
import logging
import av.container

import numpy as np
import cv2
import torch
import torchaudio

from transformers import HubertForSequenceClassification

from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

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

        # TODO: Initialize OpenCV classifiers and Hugging Face models

    
    async def recv(self):
        # Retrieve the next input frame
        frame: VideoFrame = await self.track.recv()
        img: np.ndarray = frame.to_ndarray(format="bgr24")

        # TODO: Detect if a face is present in the image

        # TODO: Crop and reshape the bounding box to 224 x 224

        # TODO: Pass face image into vision transformer model

        # TODO: Compute saliency map for face

        # TODO: Draw bounding box and saliency map onto image

        # Reconstruct and return the new frame
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
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
        self.hubert = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-er")


        AudioAnalyzer.instance = self

    def run_audio_analysis(self, filename: str):
        # TODO: load audio file
        waveform, samplerate = torchaudio.load(filename)
        resampler = torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=16000)
        waveform = waveform[0:1]
        waveform: torch.Tensor = resampler.forward(waveform)

        # TODO: obtain initial diagnosis

        # TODO: convert to spectrogram

        # TODO: compute saliency map

        return {
            "waveform": [],
            "spectrogram": [],
            "saliency": [],
            "logits": [],
            "labels": []
        }

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
    analysis = AudioAnalyzer.instance.run_audio_analysis(saved_filename)
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
            custom_recorder.addTrack(track)
        
        elif track.kind == "video":
            pc.addTrack(VideoTransformTrack(
                relay.subscribe(track), 
                transform=params["video_transform"]
            ))

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