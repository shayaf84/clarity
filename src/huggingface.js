import {
    pipeline
} from '@xenova/transformers';

const modelId = "superb/hubert-base-superb-er";
const taskId = "audio-classification";

async function initClassifier() {
    const classifier = await pipeline(taskId, modelId);

    return classifier;
}