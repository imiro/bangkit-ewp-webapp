
import * as tf from '@tensorflow/tfjs';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// import {version} from '@tensorflow/tfjs-backend-wasm/dist/version';
import { IMAGE_SIZE, loadModel, predict } from './lib';

// ### event bindings?

// ###

// --- begin load model

// set tf.js Wasm path

// configure tf.js to use wasm

// wait for it to complete

// try to load model

// wait for it to complete, then

// hide loading animation + show button

// bind click to predict event

// --- complete

(async function () {
    tfjsWasm.setWasmPath(`https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@2.0.0/dist/tfjs-backend-wasm.wasm`)
    await tf.setBackend('wasm')

    await loadModel()
    console.log('model loaded')

    document.getElementById('imgfile')
            .addEventListener('change', imgfileChangeHandler)

    hideModelLoading()
    showUploadButton()
})() // immediately run

function imgfileChangeHandler(evt) {

    let files = evt.target.files;
    // Display thumbnails & issue call to predict the image.
    for (let i = 0, f; f = files[i]; i++)
    {
      // Only process image files (skip non image files)
      if (!f.type.match('image.*')) {
        // TODO alert?
        continue;
      }

      let reader = new FileReader();
      reader.onload = onReaderLoad;

      // Read in the image file as a data URL.
      reader.readAsDataURL(f);
    }

    function onReaderLoad(e) {
      // Fill the image & call predict.
      let img = document.getElementById('placeholder')
      img.src = e.target.result;
      // img.width = IMAGE_SIZE; img.height = IMAGE_SIZE;
      img.onload = function()
      {
        console.log('image loaded, running prediction')
        let imgElem = document.getElementById('placeholder')
        let predictElem = document.getElementById('prediction-result')

        let resultSectionElem = document.getElementById('result-section');
        resultSectionElem.style.display='block';
        resultSectionElem.querySelector('.loading').style.display='block';
        predictElem.innerText = '';
        predict(imgElem).then(p => {
          // display prediction result
          resultSectionElem.querySelector('.loading').style.display='none';
          predictElem.innerText = p;
        })
      }
    };
}

function hideModelLoading() {
  document.getElementById('status').style.display='none';
}

function showUploadButton() {
  document.getElementById('imgfile-container').style.display='table';
}

// (standby)
