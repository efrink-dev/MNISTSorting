function start() {
    const loader = new ImageLoader();
    let stopCalculation = true;

    loader.imagesLoaded.then(() => {
        document.getElementById("startButton").addEventListener("click", () => {
            learnMNIST(loader);
        })
    })

}



async function learnMNIST(loader) {


    const gridCanvas = document.getElementById("testGrid");
    const gridCTX = gridCanvas.getContext("2d");


    let layers = [28 * 28, 20, 10, 2];
    const net = new NeuralNet(layers, 0.05);

    const epocs = 60000;
    const testFreq = 600;

    const modEpocs = epocs / testFreq;

    for (let trainNum = 0; trainNum <= epocs; trainNum++) {
        const nextSet = loader.getNextTraining();

        net.train([nextSet.inputs], [nextSet.expectedOut]);
        if (trainNum % modEpocs == 0) {
            console.log('Training...  ' + trainNum)
            gridCTX.fillStyle = 'white'
            gridCTX.fillRect(0, 0, 1028, 1028)
            gridCTX.strokeStyle = 'black';
            gridCTX.lineWidth = 4;
            gridCTX.moveTo(333, 0);
            gridCTX.lineTo(333, 1028);
            gridCTX.moveTo(666, 1028);
            gridCTX.lineTo(666, 0);
            gridCTX.moveTo(0, 333);
            gridCTX.lineTo(1028, 333);
            gridCTX.moveTo(0, 666);
            gridCTX.lineTo(1028, 666);


            let numRatios = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]];
            for (let i = 0; i < 100; i++) {
                const test = loader.getNextTesting();
                const output = net.test([test.inputs]);
                const gridX = output[0][1] * 1000;
                const gridY = output[0][0] * 1000;
                let position = 0;
                if (gridX <= 333) {
                    if (gridY <= 333) {
                        position = 1;
                    } else if (gridY <= 666) {
                        position = 4;
                    } else {
                        position = 7;
                    }
                } else if (gridX <= 666) {
                    if (gridY <= 333) {
                        position = 2;
                    } else if (gridY <= 666) {
                        position = 5;
                    } else {
                        position = 8;
                    }
                } else {
                    if (gridY <= 333) {
                        position = 3;
                    } else if (gridY <= 666) {
                        position = 6;
                    } else {
                        position = 9;
                    }
                }
                numRatios[position - 1][1]++;
                if (position === output.label) {
                    numRatios[position - 1][0]++;
                }
                gridCTX.putImageData(test.grid, gridX, gridY);
            }

            gridCTX.stroke();
            console.log(numRatios);
            // The UI will update every 50 milliseconds
            await delay(50);
        }
    }

}

function learnAddition() {
    let layers = [2, 60, 8, 1];
    let net = new NeuralNet(layers, 0.1);

    let trainingDataInputs = [
        [0.1, 0.1],
        [0.2, 0.1],
        [0.1, 0.2],
        [0.3, 0.1],
        [0.1, 0.3],
        [0.1, 0.4],
        [0.3, 0.2],
        [0.5, 0.1],
        [0.2, 0.4],
        [0.3, 0.4],
        [0.5, 0.2],
        [0.5, 0.3],
        [0.4, 0.4],
        [0.2, 0.7],
        [0.6, 0.3]
    ];

    let trainingDataOutputs = [
        [0.2],
        [0.3],
        [0.3],
        [0.4],
        [0.4],
        [0.5],
        [0.5],
        [0.6],
        [0.6],
        [0.7],
        [0.7],
        [0.8],
        [0.8],
        [0.9],
        [0.9],

    ];

    // Shuffle
    for (let i = trainingDataInputs.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [trainingDataInputs[i], trainingDataInputs[j]] = [trainingDataInputs[j], trainingDataInputs[i]];
        [trainingDataOutputs[i], trainingDataOutputs[j]] = [trainingDataOutputs[j], trainingDataOutputs[i]];
    }


    let testData = [
        [0.1, 0.1],
        [0.1, 0.2],
        [0.1, 0.3],
        [0.3, 0.2],
        [0.2, 0.4],
        [0.5, 0.2],
        [0.4, 0.4],
        [0.6, 0.3]
    ];

    let epocs = 100;

    for (let i = 0; i < 10; i++) {
        for (let j = 0; j < epocs; j++) {
            net.train(trainingDataInputs, trainingDataOutputs);
        }

        let sampleOut = net.test(testData);
        console.log(sampleOut);
        //console.log(net);
    }

}


class ImageLoader {
    constructor() {
        this.currentTraining = 0;
        this.currentTesting = 0;

        this.trainingData = document.getElementById("trainingData");
        this.trainingDataCTX = this.trainingData.getContext("2d", { willReadFrequently: true });

        const trainingDataImg = new Image();
        trainingDataImg.crossOrigin = "anonymous";
        trainingDataImg.src = "./train-data.png";

        this.trainingLabels = document.getElementById("trainingLabels");
        this.trainingLabelsCTX = this.trainingLabels.getContext("2d", { willReadFrequently: true });

        const trainingLabelImg = new Image();
        trainingLabelImg.crossOrigin = "anonymous";
        trainingLabelImg.src = "./train-labels.png";

        this.testingData = document.getElementById("testingData");
        this.testingDataCTX = this.testingData.getContext("2d", { willReadFrequently: true });

        const testingDataImg = new Image();
        testingDataImg.crossOrigin = "anonymous";
        testingDataImg.src = "./test-data.png";

        this.testingLabels = document.getElementById("testingLabels");
        this.testingLabelsCTX = this.testingLabels.getContext("2d", { willReadFrequently: true });

        const testingLabelImg = new Image();
        testingLabelImg.crossOrigin = "anonymous";
        testingLabelImg.src = "./test-labels.png";

        this.imagesLoaded = new Promise((resolve, reject) => {
            let imagesToLoad = 4;  // Change this to the number of images you're loading

            function imageLoaded() {
                imagesToLoad--;
                if (imagesToLoad === 0) {
                    resolve();
                }
            }

            trainingDataImg.addEventListener("load", () => {
                this.trainingDataCTX.drawImage(trainingDataImg, 0, 0);
                trainingDataImg.style.display = "none";
                imageLoaded();
            })

            trainingLabelImg.addEventListener("load", () => {
                this.trainingLabelsCTX.drawImage(trainingLabelImg, 0, 0);
                trainingLabelImg.style.display = "none";
                imageLoaded();
            })

            testingDataImg.addEventListener("load", () => {
                this.testingDataCTX.drawImage(testingDataImg, 0, 0);
                testingDataImg.style.display = "none";
                imageLoaded();
            })

            testingLabelImg.addEventListener("load", () => {
                this.testingLabelsCTX.drawImage(testingLabelImg, 0, 0);
                testingLabelImg.style.display = "none";
                imageLoaded();
            })
        });
    }


    getNextTraining() {
        if (this.currentTraining >= (244 * 244)) {
            this.currentTraining = 0;
        }
        let x = this.currentTraining % 244;
        let y = Math.floor((this.currentTraining - x) / 244);
        let output = [];
        let consoleImg = [];
        for (let i = 0; i < 28; i++) {
            let row = [];
            for (let j = 0; j < 28; j++) {
                row.push(this.trainingDataCTX.getImageData((x * 28) + j, (y * 28) + i, 1, 1).data[1]);
                output.push(this.trainingDataCTX.getImageData((x * 28) + j, (y * 28) + i, 1, 1).data[1] / 255)
            }
            consoleImg.push(row);
        }
        //console.log('x: ' + x + ' y: ' + y);
        let labelI = this.trainingLabelsCTX.getImageData(x, y, 1, 1).data[1];
        if (labelI == 0) {
            this.currentTraining++;
            return this.getNextTraining();
        }

        let expectedOutputs = [
            [0, 0], [0, 0.5], [0, 1],
            [0.5, 0], [0.5, 0.5], [0.5, 1],
            [1, 0], [1, 0.5], [1, 1]
        ];

        this.currentTraining++;
        return {
            label: labelI,
            inputs: output,
            consoleImg: consoleImg,
            expectedOut: expectedOutputs[labelI - 1]
        };
    }

    getNextTesting() {
        if (this.currentTesting >= (10000)) {
            this.currentTesting = 0;
        }
        let x = this.currentTesting % 100;

        let y = Math.round((this.currentTesting - x) / 100);
        let output = [];
        let consoleImg = [];
        for (let i = 0; i < 28; i++) {

            let row = []
            for (let j = 0; j < 28; j++) {
                row.push(this.testingDataCTX.getImageData((x * 28) + j, (y * 28) + i, 1, 1).data[1]);
                output.push(this.testingDataCTX.getImageData((x * 28) + j, (y * 28) + i, 1, 1).data[1] / 255)
            }
            consoleImg.push(row);
        }
        let label = this.testingLabelsCTX.getImageData(x, y, 1, 1).data[1];

        if (label == 0) {
            this.currentTesting++;
            return this.getNextTesting();
        }
        let expectedOutputs = [
            [0, 0], [0, 0.5], [0, 1],
            [0.5, 0], [0.5, 0.5], [0.5, 1],
            [1, 0], [1, 0.5], [1, 1]
        ];

        this.currentTesting++;
        return {
            label: label,
            inputs: output,
            grid: this.testingDataCTX.getImageData(x * 28, y * 28, 28, 28),
            consoleImg: consoleImg,
            expectedOut: expectedOutputs[label - 1]
        };

    }

    reset() {
        this.currentTesting = 0;
        this.currentTraining = 0;
    }
}


class Neuron {


    constructor(numWeights) {
        this.weights = this.generateRandomWeights(numWeights);
        this.bias = getRandomArbitrary(-1, 1);
        this.output = 0;
        this.delta = 0;
    }

    generateRandomWeights(numWeights) {
        let output = [];
        for (let i = 0; i < numWeights; i++) {
            output.push(getRandomArbitrary(-1, 1));
        }
        return output;
    }

    process(inputs) {
        let tempOut = 0.0;
        for (let i = 0; i < inputs.length; i++) {
            tempOut += this.weights[i] * inputs[i];
        }
        tempOut += this.bias;

        this.output = this.sigmoid(tempOut);
    }

    sigmoid(z) {
        return 1 / (1 + Math.exp(-z));
    }

    derivativeSigmoid(z) {
        return z * (1 - z);
    }

    getJson() {
        return {
            bias: this.bias,
            weights: this.weights
        };
    }

}

class NeuralNet {
    constructor(layers, learnRate) {
        this.inputs = new Array(layers[0]);
        this.netLayers = this.buildHiddenLayers(layers);
        this.learnRate = learnRate;
        console.log(this.netLayers)
    }

    buildHiddenLayers(layers) {
        let output = [];
        for (let i = 1; i < layers.length; i++) {
            let layer = [];
            for (let j = 0; j < layers[i]; j++) {
                let neuron = new Neuron(layers[i - 1]);
                //console.log(neuron);
                layer.push(neuron);
            }
            output.push(layer);
        }
        return output;
    }

    train(inputs, expectedOutputs) {
        for (let i = 0; i < inputs.length; i++) {
            this.inputs = inputs[i];
            this.forwardProp();
            this.backProp(expectedOutputs[i]);
            this.updateWeightsAndBiases();
        }
    }

    test(inputs) {
        let outputs = [];
        for (let i = 0; i < inputs.length; i++) {
            let output = [];
            this.inputs = inputs[i];
            this.forwardProp();
            for (let j = 0; j < this.netLayers[this.netLayers.length - 1].length; j++) {
                output.push(this.netLayers[this.netLayers.length - 1][j].output);
            }
            outputs.push(output);
        }
        return outputs;
    }

    forwardProp() {
        let curInputs = this.inputs;
        for (let i = 0; i < this.netLayers.length; i++) {
            let curLayer = this.netLayers[i];
            for (let j = 0; j < curLayer.length - 1; j++) {
                curLayer[j].process(curInputs);
            }
            curLayer[curLayer.length - 1].process(curInputs);
            curInputs = this.setInputs(curLayer);
        }
    }

    backProp(expectedOutput) {
        for (let i = 0; i < expectedOutput.length; i++) {
            this.netLayers[this.netLayers.length - 1][i].delta = expectedOutput[i] - this.netLayers[this.netLayers.length - 1][i].output;
        }

        for (let i = this.netLayers.length - 1; i > 0; i--) {
            let curLayer = this.netLayers[i];
            let prevLayer = this.netLayers[i - 1];
            for (let e = 0; e < curLayer.length; e++) {
                for (let j = 0; j < prevLayer.length; j++) {
                    prevLayer[j].delta += curLayer[e].weights[j] * curLayer[e].delta;
                }
            }
            for (let j = 0; j < prevLayer.length; j++) {
                prevLayer[j].delta = prevLayer[j].delta * prevLayer[j].derivativeSigmoid(prevLayer[j].output);
            }
        }
    }

    updateWeightsAndBiases() {
        for (let i = 0; i < this.netLayers.length; i++) {
            for (let j = 0; j < this.netLayers[i].length; j++) {
                let prevLayer = [];
                if (i == 0) {
                    prevLayer = this.inputs;
                } else {
                    prevLayer = this.netLayers[i - 1];
                }

                for (let p = 0; p < prevLayer.length; p++) {
                    let prevOutput;
                    if (i == 0) {
                        prevOutput = prevLayer[p];
                    } else {
                        prevOutput = prevLayer[p].output;
                    }
                    this.netLayers[i][j].weights[p] += prevOutput * this.learnRate * this.netLayers[i][j].delta;
                }
                this.netLayers[i][j].bias += this.learnRate * this.netLayers[i][j].delta;
            }
        }
    }

    setInputs(layer) {
        let output = [];
        for (let i = 0; i < layer.length; i++) {
            output.push(layer[i].output);
        }
        return output;
    }
}




function getRandomArbitrary(min, max) {
    return Math.random() * (max - min) + min;
}

const delay = (delayInms) => {
    return new Promise(resolve => setTimeout(resolve, delayInms));
}

start();
//learnAddition();