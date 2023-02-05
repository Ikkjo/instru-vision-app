import { useState, useEffect, useRef } from 'react';
import * as tf from "@tensorflow/tfjs";

function App() {
	const MNETV3_URL = "https://raw.githubusercontent.com/Ikkjo/mobilenetv3-model/main/model.json"
	const INSTRU_VISION_CNN_URL = "https://raw.githubusercontent.com/Ikkjo/instru-vision-cnn-model/main/model.json"
	const [isModelLoading, setIsModelLoading] = useState(false)
	const [model, setModel] = useState(null)
	const [imageURL, setImageURL] = useState(null)
	const [results, setResults] = useState([])

	const classLabels = require("./class_labels.json")

	const imageRef = useRef()
	const imageUrlTextRef = useRef()
	const fileInputRef = useRef()

	const loadModel = async (modelUrl) => {
		setIsModelLoading(true)
		try{
			const model = await tf.loadGraphModel(modelUrl)
			// Executing model when it loads to make later predictions quicker
			model.predict(tf.zeros([1, 224, 224, 3]))
			
			setModel(model)
			setIsModelLoading(false)
		} catch (error) {
			console.log(error)
			setIsModelLoading(false)	
		}
	};

	const uploadImage = (e) => {
		const {files} = e.target
		if (files.length > 0) {
			const url = URL.createObjectURL(files[0])
			setImageURL(url)
			setResults([])
		} else {
			setImageURL(null)
			setResults([])
		}
	};

	const identify = async () => {
		imageUrlTextRef.current.value = ''
		let image = imageRef.current

		// Image preprocessing for prediction

		let tensor = await tf.browser.fromPixels(image)
			.resizeBilinear([224, 224])
			.toFloat()
			.expandDims();

		let prediction = model.predict(tensor)

		// Mapping highest value of prediction to its class and sorting

		prediction = Array.from(prediction.arraySync())[0]
			.map(function (p, i) {
				return {
					probability: p,
					classLabel: classLabels[i]
				}
			}).sort(function (a, b) {
				return b.probability - a.probability
			}).slice(0, prediction.length)
		
		setResults(prediction)
	}

	const onModelTypeChange = (e) => {
		let chosen = e.target.value
		let modelUrl = ""
		if (chosen==="mnetv3") {
			modelUrl = MNETV3_URL
		} else if (chosen==="iv-cnn") {
			modelUrl = INSTRU_VISION_CNN_URL
		}
		loadModel(modelUrl)
	}

	const onUrlTextChange = (e) => {
		setImageURL(e.target.value)
		setResults([])
	}

	const triggerUpload = () => {
		fileInputRef.current.click()
	}

	useEffect(() => {
		loadModel(MNETV3_URL)
	}, []);

	return (
		<div className="App">
			<h1>What instrument is this?</h1>
			{isModelLoading ? <h2>Model is loading...</h2> : <h2>Choose an image!</h2>}
			<div className="inputHolder">
				<select onChange={onModelTypeChange}>
					<option value="mnetv3">MobileNetV3</option>
					<option value="iv-cnn">Instru-Vision CNN</option>
				</select>
				<input type="file" accept="image/*" capture="camera"
				 className="uploadInput" onChange={uploadImage} ref={fileInputRef}/>
				<button className='uploadImage' onClick={triggerUpload}>Upload Image</button>
				<span className='or'>OR</span>
				<input type="text" placeholder='Paste image URL' ref={imageUrlTextRef} onChange={onUrlTextChange}/>
			</div>
			<div className='mainWrapper'>
				<div className='mainContent'>
					<div className='imageHolder'>
						{imageURL && <img src={imageURL} alt="Upload Preview" crossOrigin="anonymous" ref={imageRef}/>}
					</div>
					{results.length > 0 && <div className='resultsHolder'>
                        {results.map((result, index) => {
                            return (
                                <div className='result' key={result.classLabel}>
                                    <span className='name'>{result.classLabel.replaceAll('_', ' ')}</span>
                                    <span className='confidence'>Confidence level: {(result.probability * 100).toFixed(2)}% {index === 0 && <span className='bestGuess'>Best Guess</span>}</span>
                                </div>
                            )
                        })}
                    </div>}
				</div>
				{imageURL && <button className="identifyButton" onClick={identify}>Identify Image</button>}
			</div>
		</div>
	);
}

export default App;
