<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced AI Word Explorer with JSON Training Data Generation</title>
    <script src="brain.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.11.0/dist/tf.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            line-height: 1.6;
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            margin: 0;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        #output, #userInput {
            border: 1px solid #ddd;
            padding: 20px;
            margin-top: 20px;
            background-color: #fff;
            border-radius: 5px;
        }
        #output {
            max-height: 400px;
            overflow-y: auto;
        }
        #userInput {
            width: 100%;
            box-sizing: border-box;
        }
        #controls {
            margin-top: 20px;
            display: flex;
            justify-content: center;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            transition: background-color 0.3s;
            margin: 0 10px;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Advanced AI Word Explorer with JSON Training Data Generation</h1>
        <div id="controls">
            <button id="startLearning">Start Learning</button>
            <button id="stopLearning" disabled>Stop Learning</button>
        </div>
        <div id="output"></div>
        <textarea id="userInput" rows="3" placeholder="Enter your message here..."></textarea>
        <button id="sendMessage">Send</button>
    </div>

<script>
    const brainNet = new brain.NeuralNetwork({
        inputSize: 20,
        outputSize: 20,
        hiddenLayers: [20]
    });
    let tfModel;
    const outputDiv = document.getElementById('output');
    const userInputElem = document.getElementById('userInput');
    const startButton = document.getElementById('startLearning');
    const stopButton = document.getElementById('stopLearning');
    const sendButton = document.getElementById('sendMessage');
    const learnedWords = new Set();
    const definitionHistory = [];
    let isLearning = false;
    let trainingData = {};

    const SEQUENCE_LENGTH = 20;

    async function createTFModel() {
        const model = tf.sequential();
        model.add(tf.layers.dense({units: 64, activation: 'relu', inputShape: [SEQUENCE_LENGTH]}));
        model.add(tf.layers.dense({units: 32, activation: 'relu'}));
        model.add(tf.layers.dense({units: SEQUENCE_LENGTH, activation: 'sigmoid'}));

        model.compile({
            optimizer: tf.train.adam(),
            loss: 'meanSquaredError',
            metrics: ['accuracy']
        });

        return model;
    }

    function stripNonLetters(word) {
        return word.replace(/[^a-zA-Z]/g, '').toLowerCase();
    }

    function isValidWord(word) {
        return word && word !== 'undefined' && word.length > 3;
    }

    async function fetchWordData(word) {
        const cleanWord = stripNonLetters(word);
        const response = await fetch(`fetch-word-data.php?word=${encodeURIComponent(cleanWord)}`);
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        return data;
    }

    function findMostInterestingWord(text) {
        const words = text.split(/\s+/)
            .map(stripNonLetters)
            .filter(word => isValidWord(word) && !learnedWords.has(word) && !trainingData.hasOwnProperty(word));
        if (words.length === 0) return null;
        return words[Math.floor(Math.random() * words.length)];
    }

    function getWordFromPreviousDefinitions() {
        if (definitionHistory.length === 0) return getRandomWord();
        let attempts = 0;
        while (attempts < 10) {
            const randomDefinition = definitionHistory[Math.floor(Math.random() * definitionHistory.length)];
            const word = findMostInterestingWord(randomDefinition);
            if (word) return word;
            attempts++;
        }
        return getRandomWord();
    }

    function getRandomWord() {
        const validWords = Array.from(learnedWords).filter(word => isValidWord(word) && !trainingData.hasOwnProperty(word));
        if (validWords.length === 0) return 'body';
        return validWords[Math.floor(Math.random() * validWords.length)];
    }

    function encodeWord(word) {
        return word.toLowerCase().split('').map(char => char.charCodeAt(0) / 255)
            .concat(Array(SEQUENCE_LENGTH).fill(0))
            .slice(0, SEQUENCE_LENGTH);
    }

    function decodeWord(encoded) {
        return encoded.map(num => String.fromCharCode(Math.round(num * 255)))
            .join('')
            .trim();
    }

    async function trainModels(input, output) {
        // Train Brain.js model
        brainNet.train([{input, output}]);

        // Train TensorFlow.js model
        const xs = tf.tensor2d([input]);
        const ys = tf.tensor2d([output]);

        await tfModel.fit(xs, ys, {
            epochs: 5,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
                }
            }
        });

        xs.dispose();
        ys.dispose();
    }

    async function predictNextWord(word) {
        const input = tf.tensor2d([encodeWord(word)]);
        const prediction = tfModel.predict(input);
        const predictedEncoded = await prediction.data();
        input.dispose();
        prediction.dispose();
        return decodeWord(Array.from(predictedEncoded));
    }

    async function updateTrainingData(word, definition, type, relatedWords) {
        if (!trainingData[word]) {
            trainingData[word] = {
                definition: definition,
                type: type,
                related: {}
            };
        }
        relatedWords.forEach(relatedWord => {
            if (!trainingData[word].related[relatedWord]) {
                trainingData[word].related[relatedWord] = 1;
            } else {
                trainingData[word].related[relatedWord]++;
            }
        });
        await saveTrainingData();
    }

    async function loadTrainingData() {
        try {
            const response = await fetch('training.json');
            if (response.ok) {
                trainingData = await response.json();
                console.log('Training data loaded successfully');
            } else {
                console.log('No existing training data found. Starting with empty data.');
            }
        } catch (error) {
            console.error('Error loading training data:', error);
        }
    }

    async function saveTrainingData() {
        try {
            const response = await fetch('save-training-data.php', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(trainingData),
            });

            if (response.ok) {
                console.log('Training data saved successfully');
            } else {
                console.error('Failed to save training data');
            }
        } catch (error) {
            console.error('Error saving training data:', error);
        }
    }

    async function learnWord(word) {
        if (!isValidWord(word) || trainingData.hasOwnProperty(word)) {
            console.log(`Invalid or duplicate word encountered: ${word}. Choosing a new word.`);
            return getWordFromPreviousDefinitions();
        }

        const cleanWord = stripNonLetters(word);
        try {
            const { definition, wiki, type } = await fetchWordData(cleanWord);
            outputDiv.innerHTML += `<p><strong>${cleanWord}</strong> (${type}):<br>Definition: ${definition}<br>Wiki: ${wiki || 'Not available'}</p>`;

            const combinedText = `${definition} ${wiki || ''}`;
            definitionHistory.push(combinedText);
            if (definitionHistory.length > 50) definitionHistory.shift();

            const relatedWords = combinedText.split(/\s+/).map(stripNonLetters).filter(isValidWord);
            await updateTrainingData(cleanWord, definition, type, relatedWords);
            learnedWords.add(cleanWord);

            const interestingWord = findMostInterestingWord(combinedText);

            if (interestingWord) {
                const input = encodeWord(cleanWord);
                const output = encodeWord(interestingWord);
                await trainModels(input, output);
                return interestingWord;
            }
        } catch (error) {
            outputDiv.innerHTML += `<p>Error learning "${cleanWord}": ${error.message}</p>`;
        }
        return getWordFromPreviousDefinitions();
    }

    async function generateResponse(input) {
        const words = input.split(/\s+/).map(stripNonLetters).filter(isValidWord);
        let response = '';
        for (const word of words) {
            if (trainingData[word]) {
                const relatedWords = Object.keys(trainingData[word].related);
                if (relatedWords.length > 0) {
                    const randomRelated = relatedWords[Math.floor(Math.random() * relatedWords.length)];
                    response += `${trainingData[word].definition} (${trainingData[word].type}) ${randomRelated} `;
                } else {
                    response += `${trainingData[word].definition} (${trainingData[word].type}) `;
                }
            } else {
                const learnedWord = await learnWord(word);
                response += learnedWord + ' ';
            }
        }
        return response.trim();
    }

    async function continuousLearning(startWord) {
        let currentWord = isValidWord(startWord) ? startWord : 'body';
        while (isLearning) {
            try {
                const newWord = await learnWord(currentWord);
                if (newWord) {
                    currentWord = newWord;
                } else {
                    currentWord = await predictNextWord(currentWord);
                }
            } catch (error) {
                console.error("Learning error:", error);
                outputDiv.innerHTML += `<p>Error occurred. Picking a word from previous definitions.</p>`;
                currentWord = getWordFromPreviousDefinitions();
            }
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    }

    window.addEventListener('load', async () => {
        await loadTrainingData();
    });

    startButton.addEventListener('click', async () => {
        if (!isLearning) {
            await loadTrainingData(); // Reload data when starting learning
            tfModel = await createTFModel();
            isLearning = true;
            startButton.disabled = true;
            stopButton.disabled = false;
            learnedWords.add('spear');
            continuousLearning('spear');
        }
    });

    stopButton.addEventListener('click', async () => {
        isLearning = false;
        startButton.disabled = false;
        stopButton.disabled = true;
        await saveTrainingData();
    });

    sendButton.addEventListener('click', async () => {
        const userMessage = userInputElem.value.trim();
        if (userMessage) {
            outputDiv.innerHTML += `<p><strong>You:</strong> ${userMessage}</p>`;
            const response = await generateResponse(userMessage);
            outputDiv.innerHTML += `<p><strong>AI:</strong> ${response}</p>`;
            userInputElem.value = '';
        }
    });
</script>
</body>
</html>