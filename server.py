"""
routes:
 - webrtc connection
 - start recording = simply toggles boolean variable
 - stop recording = toggles off, closes container, saves file, loads, runs analysis with pytorch, returns analysis
"""

import argparse
import asyncio
import json
import logging
import os
import uuid

import av.container
import cv2

import queue

import torch
import torchaudio
import numpy as np
from test_ml import run_analysis

from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder, MediaRelay, MediaRecorderContext, MediaStreamError, MediaBlackhole

import av
from av.audio.frame import AudioFrame
from av.video.frame import VideoFrame

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

is_recording = False
custom_recorder = None

class VideoTransformTrack(MediaStreamTrack):
    """
    Code used for video processing, i.e. face bounding box, fatigue detection, etc.
    """
    
    kind = "video"

    def __init__(self, track, transform):
        super().__init__()
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()

        # edge detection
        img = frame.to_ndarray(format="bgr24")
        img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

        # rebuild a VideoFrame, preserving timing information
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base

        return new_frame

class CustomMediaRecorder:
    def __init__(self):
        self.counter = 0

        container, filename = self.make_container()

        self.container = container
        self.filename = filename

        self.recording = False

        self.track = None
    
    def start_recording(self):
        self.recording = True

    def stop_recording(self):
        self.container.close()
        old_filename = self.filename

        container, filename = self.make_container()
        self.container = container
        self.filename = filename

        self.addTrack(self.track)

        self.recording = False

        return old_filename

    def make_container(self):
        filename = f"recording{self.counter:03}.mp3"
        container = av.open(file=filename, mode="w")
        self.counter += 1
        return container, filename
    
    def addTrack(self, track: MediaStreamTrack) -> None:
        """
        Add a track to be recorded.

        :param track: A :class:`aiortc.MediaStreamTrack`.
        """

        if track.kind == "audio":
            if self.container.format.name in ("wav", "alsa", "pulse"):
                codec_name = "pcm_s16le"
            elif self.container.format.name == "mp3":
                codec_name = "mp3"
            else:
                codec_name = "aac"
            stream = self.container.add_stream(codec_name)
        else:
            if self.container.format.name == "image2":
                stream = self.container.add_stream("png", rate=30)
                stream.pix_fmt = "rgb24"
            else:
                stream = self.container.add_stream("libx264", rate=30)
                stream.pix_fmt = "yuv420p"
        
        self.track = track
        self.context = MediaRecorderContext(stream)

    async def start(self) -> None:
        """
        Start recording.
        """
        if self.context.task is None:
            self.context.task = asyncio.ensure_future(self.__run_track(self.track, self.context))
    
    async def stop(self) -> None:
        """
        Stop recording.
        """
        for i, container in enumerate(self.containers):
            context = self.track
            if context.task is not None:
                context.task.cancel()
                context.task = None
                for packet in context.stream.encode(None):
                    container.mux(packet)
            
            self.track = None
            self.context = None

            container.close()

            self.container = None

    async def __run_track(
        self, track: MediaStreamTrack, context: MediaRecorderContext
    ) -> None:
        while True:
            frame = await track.recv()

            if not context.started:
                # adjust the output size to match the first frame
                context.started = True

            if self.recording:
                for packet in context.stream.encode(frame):
                    self.container.mux(packet)
                       
class AudioConsumer(MediaStreamTrack):
    """
    Code used for audio processing
    """

    kind = "audio"
    
    def __init__(self, track):
        super().__init__()
        self.track = track
        self.accumulated_frames = []
        self.counter = 0
    
    async def recv(self):
        frame: AudioFrame = await self.track.recv()
        """
        samples = frame.to_ndarray()
        
        rate = frame.sample_rate

        self.accumulated_frames += [samples]
        if len(self.accumulated_frames) > 100:
            self.accumulated_frames.pop(0)

        self.counter += 1
        if self.counter % 200 == 0:
            all_samples = np.concatenate(self.accumulated_frames, axis=1)
            print(f"all samples are {all_samples.shape}")

            all_samples = torch.from_numpy(all_samples)
            torchaudio.save("test.wav", all_samples, 48000)

        # todo: keep a list of the last N seconds of audio
        #       to continually feed into transformer model

        print(f"received audio frame: {samples.shape} at {rate} Hz")
        """
        return frame
    
async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)

async def start_recording(request):
    # TODO: tell the media recorder to START
    custom_recorder.start_recording()
    return web.Response(content_type="application/json", text="{\"success\": \"true\"}")

async def stop_recording(request):
    # TODO: tell media recorder to STOP, then wait for filename, then process w pytorch, then return goodies
    filename = custom_recorder.stop_recording()
    analysis = run_analysis(filename)
    payload = json.dumps(analysis)
    return web.Response(content_type="application/json", text=payload)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    # prepare local media
    #player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
    #if args.record_to:
    #filename_queue = queue.Queue()
    #recorder = CustomMediaRecorder(filename_queue)
    #else:
    global custom_recorder
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
            #pc.addTrack(player.audio)
            custom_recorder.addTrack(track)
            pc.addTrack(
                AudioConsumer(relay.subscribe(track))
            )
            # this is where we need to consume audio track
        elif track.kind == "video":
            pc.addTrack(
                VideoTransformTrack(
                    relay.subscribe(track), transform=params["video_transform"]
                )
            )
            #if args.record_to:
            #    recorder.addTrack(relay.subscribe(track))

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
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    
    parser.add_argument("--verbose", "-v", action="count", default=True)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_get("/stop", stop_recording)
    app.router.add_post("/start", start_recording)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )