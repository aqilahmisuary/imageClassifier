/*
Classifications
- Gyan Mudra
- Shuni Mudra
- Surya Mudra
- Buddhi Mudra
- Stop
*/

var audio;

const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');
let model;
let str;
var allText;

var sound_01 = new Howl({
  src: ['./audio/audio_01.wav'],
  format: ['wav']
});

var sound_02 = new Howl({
  src: ['./audio/audio_02.wav'],
  format: ['wav']
});

var sound_03 = new Howl({
  src: ['./audio/audio_03.wav'],
  format: ['wav']
});

var sound_04 = new Howl({
  src: ['./audio/audio_04.wav'],
  format: ['wav']
});


sound_01.volume(0.1);
sound_02.volume(0.1);
sound_03.volume(0.1);
sound_04.volume(0.1);

document.getElementById('start').addEventListener('click', () => sound.play());


function load() {
  // Load dataset file into a fresh classifier:
  classifier.setClassifierDataset( Object.fromEntries( JSON.parse(allText).map(([label, data, shape])=>[label, tf.tensor(data, shape)]) ) );

 }

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  model = await mobilenet.load();
  console.log('Successfully loaded model');

  // Create an object from Tensorflow.js data API which could capture image 
  // from the web camera as Tensor.
  const webcam = await tf.data.webcam(webcamElement);

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = async classId => {
    // Capture an image from the web camera.
    const img = await webcam.capture();

    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = model.infer(img, true);

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);

    // Dispose the tensor to release the memory.
    img.dispose();
  };

  function save() {
    // Save it to a string:
    str = JSON.stringify( Object.entries(classifier.getClassifierDataset()).map(([label, data])=>[label, Array.from(data.dataSync()), data.shape]) );
    console.log("Dataset Saved");
  }

  // When clicking a button, add an example for that class.
  document.getElementById('class-gyan').addEventListener('click', () => addExample(0));
  document.getElementById('class-shuni').addEventListener('click', () => addExample(1));
  document.getElementById('class-surya').addEventListener('click', () => addExample(2));
  document.getElementById('class-buddhi').addEventListener('click', () => addExample(3));
  document.getElementById('class-nothing').addEventListener('click', () => addExample(4));

  //save dataset
  document.getElementById('save').addEventListener('click', () => save());

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();

      // Get the activation from mobilenet from the webcam.
      const activation = model.infer(img, 'conv_preds');
      // Get the most likely class and confidence from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['Gyan', 'Shuni', 'Surya', `Buddhi`, `Nothing`];
      document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `;

      if(classes[result.label] == 'Gyan'){
        sound_01.play();
      } else if(classes[result.label] == 'Shuni') {
        sound_02.play();
      } else if(classes[result.label] == 'Surya') {
        sound_03.play();
      } else if(classes[result.label] == 'Buddhi') {
        sound_04.play();
      }

      // Dispose the tensor to release the memory.
      img.dispose();
    }

    await tf.nextFrame();
  } 
}

//download textfile
function download(filename, text) {
  var element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);

  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
}

// Start file download.
document.getElementById("dwn-btn").addEventListener("click", function(){
  // // Generate download of hello.txt file with some content
  // var text = document.getElementById("text-val").value;
 var filename = "dataset.txt";
  
  download(filename, str);
}, false);

//read textfile

function readTextFile(file)
{
    var rawFile = new XMLHttpRequest();
    rawFile.open("GET", file, false);
    rawFile.onreadystatechange = function ()
    {
        if(rawFile.readyState === 4)
        {
            if(rawFile.status === 200 || rawFile.status == 0)
            {
                allText = rawFile.responseText;
                
                //alert(allText);
            }
        }
    }
    rawFile.send(null);
}


//Read dataset
readTextFile("dataset.txt");
//Load dataset into classifier
load();
//Run classifier app
app();
