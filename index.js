let net;
const webcamElement = document.getElementById('webcam');
const multAddAction = document.getElementById('multAddAction');
const labelA = document.getElementById('label-a');
const labelB = document.getElementById('label-b');
const labelC = document.getElementById('label-c');
const classifier = knnClassifier.create();

async function setupWebcam() {
    return new Promise((resolve, reject) => {
        const navigatorAny = navigator;
        navigator.getUserMedia = navigator.getUserMedia ||
            navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
            navigatorAny.msGetUserMedia;
        if (navigator.getUserMedia) {
            navigator.getUserMedia({ video: true },
                stream => {
                    webcamElement.srcObject = stream;
                    webcamElement.addEventListener('loadeddata', () => resolve(), false);
                },
                error => reject());
        } else {
            reject();
        }
    });
}

async function app() {
    console.log('Loading mobilenet..');

    // Load the model.
    net = await mobilenet.load();
    console.log('Sucessfully loaded model');

    await setupWebcam();

    // Reads an image from the webcam and associates it with a specific class
    // index.
    const addExample = classId => {
        var numbAddActions = multAddAction.value;
        console.log(numbAddActions);
        var count = 0;
        do {
            // Get the intermediate activation of MobileNet 'conv_preds' and pass that
            // to the KNN classifier.
            resizeVideo(224);
            const activation = net.infer(webcamElement, 'conv_preds');
            // Pass the intermediate activation to the classifier.
            classifier.addExample(activation, classId);
            count++;
            console.log("Add more one element");
        } while (count < numbAddActions);
        resizeVideo(500);
    };

    const resizeVideo = resizePixel => {
        webcamElement.style.width = resizePixel;
        webcamElement.style.height = resizePixel;
    }

    // When clicking a button, add an example for that class.
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));

    while (true) {
        if (classifier.getNumClasses() > 0) {
            // Get the activation from mobilenet from the webcam.
            const activation = net.infer(webcamElement, 'conv_preds');
            // Get the most likely class and confidences from the classifier module.
            const result = await classifier.predictClass(activation);

            const classes = [labelA.value, labelB.value, labelC.value];
            document.getElementById('console').innerText = `
          Prediction: ${classes[result.classIndex]}
          Probability: ${result.confidences[result.classIndex]}
        `;
        }

        await tf.nextFrame();
    }
}

app();