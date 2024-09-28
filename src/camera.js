
/* todo: load camera stream, downsample in both spatial and temporal axes */

/* main idea: capture video frames from video element and send to server along with audio */

async function getMedia() {

    let stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: { width: 1280, height: 720 }
    });

}