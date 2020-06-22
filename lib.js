import * as tf from '@tensorflow/tfjs';
// import modelFile from 'url:./model/model.json'
// import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// import * as cam from './cam';
// import jimp from 'jimp';

// const MODEL_PATH = modelFile;
const MODEL_PATH = 'model/model.json';
// const MODEL_PATH = 'https://firebasestorage.googleapis.com/v0/b/catch-of-the-day-45cb7.appspot.com/o/model%2Fmodel.json?alt=media';

export const IMAGE_SIZE = 224;

let model;
let labels;

export const loadModel = async () => {
  console.log('loading model....');

  // mobilenet
  console.log('loading mobilenetv1')
  model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json', {});

  let response = await fetch('https://raw.githubusercontent.com/leferrad/tensorflow-mobilenet/master/imagenet/labels.txt')
  let tresp = await response.text()
  labels = tresp.split('\n')
  model.summary(); // TODO remove from production
  return;
  // await tf.setBackend('wasm');
  model = await tf.loadLayersModel(MODEL_PATH, {});

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1])).dispose();

  // Show model layers and summary
  model.summary();
};

export const clearModelMemory = () => {
  model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 1])).dispose();
};

/**
 * Given an image element, makes a prediction through model returning the
 * probabilities of the top K classes.
 */
export async function predict(imgElement) {
  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits =  tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    // const img = tf.browser.fromPixels(imgElement, 1);

    // Normalize the image from [0, 225] to [INPUT_MIN, INPUT_MAX]
    const normalizationConstant = 1.0 / 255.0;
    // const normalized = img.toFloat().mul(normalizationConstant);

    let img = tf.browser.fromPixels(imgElement)
      .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE], false)
      .expandDims(null)
      .toFloat()
      .div(tf.scalar(127)).sub(tf.scalar(1))
      // .mul(normalizationConstant) // TODO activate in production

    // const image = tf.image.resizeBilinear(normalized, [IMAGE_SIZE, IMAGE_SIZE], false);

    // Reshape to a single-element batch so we can pass it to predict.
    // const batched = image.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 1]);

    startTime2 = performance.now();

    // Make a prediction through model.
    return model.predict(img);
  });

  // Convert logits to probabilities and class names.
  // NOTE i think this is a 'hack' as tf.tidy() does not allow promises as return value
  const classes = await logits.data();

  // console.log('Predictions: ', classes);
  console.log('predicted class')
  logits.as1D().argMax().print()

  const predictedClass = logits.as1D().argMax().dataSync()[0] + 1;
  // const labelName = classes[predictedClass] < 0.5? 'NORMAL' : 'PNEUMONIA';
  const labelName = labels[predictedClass];
  const probability = classes[predictedClass];

  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  console.log(`Done in ${Math.floor(totalTime1)} ms ` +
      `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

  // await new Promise(r => setTimeout(r, 2500));
  // Return the best probability label
  return `${labelName} (${Math.floor(probability * 100)}%)`;
}

export function resizeImage(imgElement){
  const canvas = document.createElement('canvas');
  canvas.height=IMAGE_SIZE;
  canvas.width=IMAGE_SIZE;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height); // clear canvas
  ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);

  const img = document.createElement('img');
  img.src = canvas.toDataURL("image/jpeg");;
  img.width = IMAGE_SIZE;
  img.height = IMAGE_SIZE;

  return img;
}


export const classNames = ['Alfalfa', 'Asparagus', 'Blue Vervain', 'Broadleaf Plantain', 'Bull Thistle', 'Cattail', 'Chickweed', 'Chicory', 'Cleavers', 'Coltsfoot', 'Common Sow Thistle', 'Common Yarrow', 'Coneflower', 'Creeping Charlie', 'Crimson Clover', 'Curly Dock', 'Daisy Fleabane', 'Dandellion', 'Downy Yellow Violet', 'Elderberry', 'Evening Primrose', 'Fern Leaf Yarrow', 'Field Pennycress', 'Fireweed', 'Forget Me Not', 'Garlic Mustard', 'Harebell', 'Henbit', 'Herb Robert', 'Japanese Knotweed', 'Joe Pye Weed', 'Knapweed', 'Kudzu', 'Lambs Quarters', 'Mallow', 'Mayapple', 'Meadowsweet', 'Milk Thistle', 'Mullein', 'New England Aster', 'Partridgeberry', 'Peppergrass', 'Pickerelweed', 'Pineapple Weed', 'Prickly Pear Cactus', 'Purple Deadnettle', 'Queen Annes Lace', 'Red Clover', 'Sheep Sorrel', 'Shepherds Purse', 'Spring Beauty', 'Sunflower', 'Supplejack Vine', 'Tea Plant', 'Teasel', 'Toothwort', 'Vervain Mallow', 'Wild Bee Balm', 'Wild Black Cherry', 'Wild Grape Vine', 'Wild Leek', 'Wood Sorrel'];

window.loadModel = loadModel;
window.clearModelMemory = clearModelMemory;
window.predict = predict;
window.resizeImage = resizeImage;
window.tf = tf;
