/* LOGIC FOR NORMAL BROWSER UTILITY */



/* LOGIC FOR UPDATING CANVAS ELEMENTS */

/** @type {HTMLCanvasElement} */
const canvas = document.getElementById("canvas");

function resizeCanvasToDisplaySize() {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    console.log(`Resized canvas to ${canvas.width} by ${canvas.height}`)
}

resizeCanvasToDisplaySize();

const ctx = canvas.getContext("2d");
ctx.fillStyle = "black";
ctx.fillRect(100, 100, 200, 200);

/* LOGIC FOR WEBRTC VIDEO AND AUDIO STREAMING */

class WebRTCManager {
    /**
     * @type {WebRTCManager} Saved instance
     */
    static instance;

    /**
     * Construct a new WebRTC manager. Initializes the RTCPeerConnection object
     * and prepares the callbacks necessary to direct incoming video streams
     * to the corresponding <video> element.
     */
    constructor() {
        this.pc = new RTCPeerConnection({
            sdpSemantics: "unified-plan",
            iceServers: [
                { urls: ['stun:stun.l.google.com:19302']}
            ]
        });

        this.pc.addEventListener("track", event => {
            if (event.track.kind == "video") {
                document.getElementById("video").srcObject = event.streams[0];
            }
        });

        WebRTCManager.instance = this;
    }

    /**
     * Start the WebRTC video stream
     */
    async start() {

        const userMedia = await navigator.mediaDevices.getUserMedia({
            audio: true,
            video: true
        });

        const tracks = userMedia.getTracks();

        for (let i = 0; i < tracks.length; i ++) {
            this.pc.addTrack(tracks[i], userMedia);
        }

        const offer = await this.pc.createOffer();

        await this.pc.setLocalDescription(offer);

        await new Promise(resolve => {
            if (this.pc.iceGatheringState == "complete") {
                resolve()
            } else {
                const iceStateChangeListener = () => {
                    if (this.pc.iceGatheringState == "complete") {
                        this.pc.removeEventListener("icegatheringstatechange", iceStateChangeListener);
                        resolve();
                    }
                };
                this.pc.addEventListener("icegatheringstatechange", iceStateChangeListener);
            }
        });

        const description = this.pc.localDescription;
        
        console.log("Sending offer");

        const response = await fetch("/offer", {
            body: JSON.stringify({
                sdp: description.sdp,
                type: description.type
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });

        const responseJson = await response.json();

        await this.pc.setRemoteDescription(responseJson);
    }

    /**
     * Stop the WebRTC video stream
     */
    async stop() {
        if (this.pc.getTransceivers) {
            this.pc.getTransceivers().forEach((transceiver) => {
                if (transceiver.stop) {
                    transceiver.stop();
                }
            });
        }

        this.pc.getSenders().forEach(sender => sender.track.stop());

        setTimeout(() => this.pc.close(), 500);
    }
}

/* Let's light this candle! */

(new WebRTCManager()).start().then(() => console.log("Connected!"));