<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced AI Word Explorer</title>
    <script src="brain.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.11.0/dist/tf.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #4CAF50;
            --hover-color: #45a049;
            --disabled-color: #cccccc;
            --shadow-color: rgba(0,0,0,0.1);
            --background-gradient: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Roboto', sans-serif;
            line-height: 1.6;
            min-height: 100vh;
            background: var(--background-gradient);
            padding: 20px;
            margin: 0;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 10px 20px var(--shadow-color);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            color: #333;
            font-size: 2rem;
            margin-bottom: 10px;
        }

        .stats-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }

        .stat-item {
            text-align: center;
            padding: 10px;
            background: white;
            border-radius: 6px;
            box-shadow: 0 2px 4px var(--shadow-color);
        }

        .stat-label {
            font-size: 0.9rem;
            color: #666;
        }

        .stat-value {
            font-size: 1.2rem;
            font-weight: bold;
            color: var(--primary-color);
        }

        .controls {
            display: flex;
            gap: 10px;
            margin: 20px 0;
            justify-content: center;
        }

        .button {
            padding: 10px 20px;
            font-size: 1rem;
            cursor: pointer;
            background-color: var(--primary-color);
            color: white;
            border: none;
            border-radius: 6px;
            transition: all 0.3s ease;
        }

        .button:hover:not(:disabled) {
            background-color: var(--hover-color);
            transform: translateY(-1px);
        }

        .button:disabled {
            background-color: var(--disabled-color);
            cursor: not-allowed;
        }

        .output-container {
            height: 400px;
            overflow-y: auto;
            padding: 15px;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 6px;
            margin-bottom: 20px;
        }

        .word-entry {
            margin-bottom: 15px;
            padding: 10px;
            border-left: 4px solid var(--primary-color);
            background: #f8f9fa;
        }

        .word-entry h3 {
            color: #333;
            margin-bottom: 5px;
        }

        .word-entry p {
            margin: 5px 0;
            color: #666;
        }

        .input-container {
            position: relative;
        }

        .input-container textarea {
            width: 100%;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 6px;
            resize: vertical;
            min-height: 100px;
            font-family: inherit;
            margin-bottom: 10px;
        }

        .memory-warning {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px;
            background: #ff4444;
            color: white;
            border-radius: 6px;
            display: none;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .error-message {
            color: #ff4444;
            margin: 10px 0;
            padding: 10px;
            background: #fff;
            border-left: 4px solid #ff4444;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Advanced AI Word Explorer</h1>
            <p>Exploring language through AI-powered learning</p>
        </div>

        <div class="stats-panel">
            <div class="stat-item">
                <div class="stat-label">Words Learned</div>
                <div class="stat-value" id="wordsLearnedCount">0</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Memory Usage</div>
                <div class="stat-value" id="memoryUsage">0 MB</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Learning Rate</div>
                <div class="stat-value" id="learningRate">0/min</div>
            </div>
        </div>

        <div class="controls">
            <button id="startLearning" class="button">Start Learning</button>
            <button id="stopLearning" class="button" disabled>Stop Learning</button>
            <button id="exportData" class="button">Export Data</button>
        </div>

        <div id="output" class="output-container"></div>

        <div class="input-container">
            <textarea id="userInput" placeholder="Enter your message here..." rows="4"></textarea>
            <button id="sendMessage" class="button">Send Message</button>
        </div>
    </div>

    <div class="memory-warning" id="memoryWarning">
        High memory usage detected! Consider saving and refreshing.
    </div>

    <script>
        // Constants
        const CONFIG = {
            MAX_TRAINING_ENTRIES: 1000,
            MAX_RELATED_WORDS: 50,
            BATCH_SIZE: 100,
            SEQUENCE_LENGTH: 20,
            MEMORY_WARNING_THRESHOLD: 0.8, // 80% of available memory
            LEARNING_INTERVAL: 1000, // 1 second
        };

        // State management
        const state = {
            isLearning: false,
            trainingData: {},
            learnedWords: new Set(),
            definitionHistory: [],
            learningStartTime: null,
            wordsLearnedThisSession: 0
        };

        // Neural Network initialization
        class AIManager {
            constructor() {
                this.brainNet = new brain.NeuralNetwork({
                    inputSize: CONFIG.SEQUENCE_LENGTH,
                    outputSize: CONFIG.SEQUENCE_LENGTH,
                    hiddenLayers: [CONFIG.SEQUENCE_LENGTH]
                });
                this.tfModel = null;
            }

            async initTFModel() {
                const model = tf.sequential();
                model.add(tf.layers.dense({
                    units: 64,
                    activation: 'relu',
                    inputShape: [CONFIG.SEQUENCE_LENGTH]
                }));
                model.add(tf.layers.dense({
                    units: 32,
                    activation: 'relu'
                }));
                model.add(tf.layers.dense({
                    units: CONFIG.SEQUENCE_LENGTH,
                    activation: 'sigmoid'
                }));

                model.compile({
                    optimizer: tf.train.adam(),
                    loss: 'meanSquaredError',
                    metrics: ['accuracy']
                });

                this.tfModel = model;
            }

            async predict(word) {
                const input = tf.tensor2d([this.encodeWord(word)]);
                const prediction = this.tfModel.predict(input);
                const predictedEncoded = await prediction.data();
                input.dispose();
                prediction.dispose();
                return this.decodeWord(Array.from(predictedEncoded));
            }

            encodeWord(word) {
                return word.toLowerCase().split('')
                    .map(char => char.charCodeAt(0) / 255)
                    .concat(Array(CONFIG.SEQUENCE_LENGTH).fill(0))
                    .slice(0, CONFIG.SEQUENCE_LENGTH);
            }

            decodeWord(encoded) {
                return encoded
                    .map(num => String.fromCharCode(Math.round(num * 255)))
                    .join('')
                    .trim();
            }

            async train(input, output) {
                this.brainNet.train([{ input, output }]);
                const xs = tf.tensor2d([input]);
                const ys = tf.tensor2d([output]);

                await this.tfModel.fit(xs, ys, {
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
        }

        // Memory management
        class MemoryManager {
            constructor() {
                this.lastCheck = Date.now();
            }

            checkMemory() {
                if (Date.now() - this.lastCheck < 1000) return;
                this.lastCheck = Date.now();

                if (window.performance && window.performance.memory) {
                    const memoryInfo = window.performance.memory;
                    const usedMemory = memoryInfo.usedJSHeapSize;
                    const totalMemory = memoryInfo.jsHeapSizeLimit;
                    const memoryUsage = usedMemory / totalMemory;

                    document.getElementById('memoryUsage').textContent = 
                        `${Math.round(usedMemory / 1024 / 1024)} MB`;

                    if (memoryUsage > CONFIG.MEMORY_WARNING_THRESHOLD) {
                        this.showMemoryWarning();
                    }
                }
            }

            showMemoryWarning() {
                const warning = document.getElementById('memoryWarning');
                warning.style.display = 'block';
                setTimeout(() => {
                    warning.style.display = 'none';
                }, 5000);
            }
        }

        // Data management
        class DataManager {
            constructor() {
                this.memoryManager = new MemoryManager();
            }

            async saveTrainingData() {
                try {
                    const entries = Object.entries(state.trainingData);
                    const chunks = [];

                    for (let i = 0; i < entries.length; i += CONFIG.BATCH_SIZE) {
                        chunks.push(Object.fromEntries(entries.slice(i, i + CONFIG.BATCH_SIZE)));
                    }

                    for (const chunk of chunks) {
                        const response = await fetch('save-training-data.php', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                chunk: chunk,
                                isPartial: chunks.length > 1,
                                timestamp: Date.now()
                            }),
                        });

                        if (!response.ok) throw new Error(response.statusText);
                    }

                    console.log('Training data saved successfully');
                } catch (error) {
                    console.error('Error saving training data:', error);
                    this.handleError(error);
                }
            }

            async loadTrainingData() {
                try {
                    const response = await fetch('training.json');
                    if (!response.ok) throw new Error(response.statusText);

                    const data = await response.json();
                    this.processLoadedData(data);
                } catch (error) {
                    console.error('Error loading training data:', error);
                    state.trainingData = {};
                }
            }

            processLoadedData(data) {
                const entries = Object.entries(data);
                if (entries.length > CONFIG.MAX_TRAINING_ENTRIES) {
                    const sorted = entries.sort(([,a], [,b]) => 
                        (b.lastUpdated || 0) - (a.lastUpdated || 0));
                    state.trainingData = Object.fromEntries(
                        sorted.slice(0, CONFIG.MAX_TRAINING_ENTRIES)
                    );
                } else {
                    state.trainingData = data;
                }

                for (const word in state.trainingData) {
                    if (state.trainingData[word].related) {
                        state.trainingData[word].related = 
                            this.pruneRelatedWords(state.trainingData[word].related);
                    }
                }
            }

            pruneRelatedWords(related) {
                const entries = Object.entries(related);
                if (entries.length <= CONFIG.MAX_RELATED_WORDS) return related;
                
                const sorted = entries.sort(([,a], [,b]) => b - a);
                return Object.fromEntries(sorted.slice(0, CONFIG.MAX_RELATED_WORDS));
            }

            handleError(error) {
                const outputDiv = document.getElementById('output');
                outputDiv.innerHTML += `
                    <div class="error-message">
                        Error: ${error.message}
                    </div>
                `;
            }
        }

        // Word processing
        class WordProcessor {
            static stripNonLetters(word) {
                return word.replace(/[^a-zA-Z]/g, '').toLowerCase();
            }

            static isValidWord(word) {
                return word && word !== 'undefined' && word.length > 3;
            }

            static async fetchWordData(word) {
                const cleanWord = this.stripNonLetters(word);
                const response = await fetch(
                    `fetch-word-data.php?word=${encodeURIComponent(cleanWord)}`
                );
                const data = await response.json();
                
                if (data.error) throw new Error(data.error);
                return data;
            }

            static findMostInterestingWord(text) {
                const words = text.split(/\s+/)
                    .map(this.stripNonLetters)
                    .filter(word => this.isValidWord(word) && 
                        !state.learnedWords.has(word) && 
                        !state.trainingData.hasOwnProperty(word)
                    );
                if (words.length === 0) return null;
                return words[Math.floor(Math.random() * words.length)];
            }

            static getWordFromPreviousDefinitions() {
                if (state.definitionHistory.length === 0) return this.getRandomWord();
                let attempts = 0;
                while (attempts < 10) {
                    const randomDefinition = state.definitionHistory[
                        Math.floor(Math.random() * state.definitionHistory.length)
                    ];
                    const word = this.findMostInterestingWord(randomDefinition);
                    if (word) return word;
                    attempts++;
                }
                return this.getRandomWord();
            }

            static getRandomWord() {
                const validWords = Array.from(state.learnedWords)
                    .filter(word => this.isValidWord(word) && 
                        !state.trainingData.hasOwnProperty(word)
                    );
                if (validWords.length === 0) return 'knowledge';
                return validWords[Math.floor(Math.random() * validWords.length)];
            }
        }

        // Learning engine
        class LearningEngine {
            constructor() {
                this.aiManager = new AIManager();
                this.dataManager = new DataManager();
            }

            async init() {
                await this.aiManager.initTFModel();
                await this.dataManager.loadTrainingData();
            }

            async learnWord(word) {
                if (!WordProcessor.isValidWord(word) || state.trainingData.hasOwnProperty(word)) {
                    console.log(`Skipping invalid or known word: ${word}`);
                    return WordProcessor.getWordFromPreviousDefinitions();
                }

                const cleanWord = WordProcessor.stripNonLetters(word);
                try {
                    const { definition, wiki, type } = await WordProcessor.fetchWordData(cleanWord);
                    this.updateUI(cleanWord, definition, type, wiki);

                    const combinedText = `${definition} ${wiki || ''}`;
                    this.updateHistory(combinedText);

                    const relatedWords = combinedText.split(/\s+/)
                        .map(WordProcessor.stripNonLetters)
                        .filter(WordProcessor.isValidWord);

                    await this.updateTrainingData(cleanWord, definition, type, relatedWords);
                    state.learnedWords.add(cleanWord);
                    state.wordsLearnedThisSession++;
                    this.updateStats();

                    const interestingWord = WordProcessor.findMostInterestingWord(combinedText);
                    if (interestingWord) {
                        const input = this.aiManager.encodeWord(cleanWord);
                        const output = this.aiManager.encodeWord(interestingWord);
                        await this.aiManager.train(input, output);
                        return interestingWord;
                    }
                } catch (error) {
                    console.error('Error learning word:', error);
                    this.dataManager.handleError(error);
                }
                return WordProcessor.getWordFromPreviousDefinitions();
            }

            updateUI(word, definition, type, wiki) {
                const outputDiv = document.getElementById('output');
                outputDiv.innerHTML += `
                    <div class="word-entry">
                        <h3>${word}</h3>
                        <p><strong>Type:</strong> ${type}</p>
                        <p><strong>Definition:</strong> ${definition}</p>
                        ${wiki ? `<p><strong>Additional Info:</strong> ${wiki}</p>` : ''}
                    </div>
                `;
                outputDiv.scrollTop = outputDiv.scrollHeight;
            }

            updateHistory(text) {
                state.definitionHistory.push(text);
                if (state.definitionHistory.length > 50) {
                    state.definitionHistory.shift();
                }
            }

            updateStats() {
                document.getElementById('wordsLearnedCount').textContent = 
                    state.wordsLearnedThisSession.toString();
                
                if (state.learningStartTime) {
                    const minutes = (Date.now() - state.learningStartTime) / 60000;
                    const rate = (state.wordsLearnedThisSession / minutes).toFixed(1);
                    document.getElementById('learningRate').textContent = `${rate}/min`;
                }
            }

            async updateTrainingData(word, definition, type, relatedWords) {
                if (Object.keys(state.trainingData).length >= CONFIG.MAX_TRAINING_ENTRIES) {
                    const entries = Object.entries(state.trainingData);
                    const sorted = entries.sort(([,a], [,b]) => 
                        (b.lastUpdated || 0) - (a.lastUpdated || 0));
                    state.trainingData = Object.fromEntries(
                        sorted.slice(0, CONFIG.MAX_TRAINING_ENTRIES - 1)
                    );
                }

                state.trainingData[word] = {
                    definition,
                    type,
                    related: {},
                    lastUpdated: Date.now()
                };

                relatedWords.forEach(relatedWord => {
                    if (!state.trainingData[word].related[relatedWord]) {
                        state.trainingData[word].related[relatedWord] = 1;
                    } else {
                        state.trainingData[word].related[relatedWord]++;
                    }
                });

                state.trainingData[word].related = 
                    this.dataManager.pruneRelatedWords(state.trainingData[word].related);

                if (Object.keys(state.trainingData).length % CONFIG.BATCH_SIZE === 0) {
                    await this.dataManager.saveTrainingData();
                }
            }

            async generateResponse(input) {
                const words = input.split(/\s+/)
                    .map(WordProcessor.stripNonLetters)
                    .filter(WordProcessor.isValidWord);
                
                let response = '';
                for (const word of words) {
                    if (state.trainingData[word]) {
                        const relatedWords = Object.keys(state.trainingData[word].related);
                        if (relatedWords.length > 0) {
                            const randomRelated = relatedWords[
                                Math.floor(Math.random() * relatedWords.length)
                            ];
                            response += `${state.trainingData[word].definition} ` +
                                `(${state.trainingData[word].type}) ${randomRelated} `;
                        } else {
                            response += `${state.trainingData[word].definition} ` +
                                `(${state.trainingData[word].type}) `;
                        }
                    } else {
                        const learnedWord = await this.learnWord(word);
                        response += learnedWord + ' ';
                    }
                }
                return response.trim();
            }
        }

        // Main application
        class WordExplorer {
            constructor() {
                this.learningEngine = new LearningEngine();
                this.setupEventListeners();
            }

            async init() {
                await this.learningEngine.init();
                this.updateButtonStates();
            }

            setupEventListeners() {
                document.getElementById('startLearning').addEventListener('click', 
                    () => this.startLearning());
                document.getElementById('stopLearning').addEventListener('click', 
                    () => this.stopLearning());
                document.getElementById('sendMessage').addEventListener('click', 
                    () => this.handleUserMessage());
                document.getElementById('exportData').addEventListener('click', 
                    () => this.exportData());
            }

            updateButtonStates() {
                const startButton = document.getElementById('startLearning');
                const stopButton = document.getElementById('stopLearning');
                startButton.disabled = state.isLearning;
                stopButton.disabled = !state.isLearning;
            }

            async startLearning() {
                if (state.isLearning) return;
                
                state.isLearning = true;
                state.learningStartTime = Date.now();
                this.updateButtonStates();
                state.learnedWords.add('knowledge');
                this.continuousLearning('knowledge');
            }

            async stopLearning() {
                state.isLearning = false;
                this.updateButtonStates();
                await this.learningEngine.dataManager.saveTrainingData();
            }

            async continuousLearning(startWord) {
                let currentWord = WordProcessor.isValidWord(startWord) ? startWord : 'knowledge';
                while (state.isLearning) {
                    try {
                        const newWord = await this.learningEngine.learnWord(currentWord);
                        currentWord = newWord || await this.learningEngine.aiManager.predict(currentWord);
                    } catch (error) {
                        console.error("Learning error:", error);
                        currentWord = WordProcessor.getWordFromPreviousDefinitions();
                    }
                    await new Promise(resolve => setTimeout(resolve, CONFIG.LEARNING_INTERVAL));
                }
            }

            async handleUserMessage() {
                const userInput = document.getElementById('userInput');
                const message = userInput.value.trim();
                if (!message) return;

                const outputDiv = document.getElementById('output');
                outputDiv.innerHTML += `
                    <div class="word-entry">
                        <h3>You:</h3>
                        <p>${message}</p>
                    </div>
                `;

                const response = await this.learningEngine.generateResponse(message);
                outputDiv.innerHTML += `
                    <div class="word-entry">
                        <h3>AI:</h3>
                        <p>${response}</p>
                    </div>
                `;

                userInput.value = '';
                outputDiv.scrollTop = outputDiv.scrollHeight;
            }

            exportData() {
                const dataStr = JSON.stringify(state.trainingData, null, 2);
                const dataBlob = new Blob([dataStr], { type: 'application/json' });
                const url = URL.createObjectURL(dataBlob);
                const link = document.createElement('a');
                link.href = url;
                link.download = 'training-data.json';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
            }
        }

        // Initialize application
        window.addEventListener('load', async () => {
            const app = new WordExplorer();
            await app.init();
        });
    </script>
</body>
</html>