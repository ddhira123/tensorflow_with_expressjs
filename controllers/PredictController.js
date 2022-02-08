const tf = require('@tensorflow/tfjs-node');
const image = require('get-image-data');
const fs = require('fs');
var path = require('path');

const classes = ['rock', 'paper', 'scissors'];

exports.makePredictions = async (req, res, next) => {
    const imagePath = `./public/images/${req && req['filename']}`;
    try {
      const loadModel = async (img) => {
        const output = {};
        // laod model
        console.log('Loading.......')
        const model = await tf.node.loadSavedModel(path.join(__dirname,'..', 'SavedModel'));
        // classify
        // output.predictions = await model.predict(img).data();
        let predictions = await model.predict(img).data();
        predictions = Array.from(predictions)
                            .map((prob, idx) => {
                                var cs = ['paper', 'rock', 'scissors'];
                                return {class: cs[idx], probability: prob}
                            })
                            .sort((a, b) => b.probability - a.probability)[0];
        output.success = true;
        output.message = `Success.`;
        output.predictions = predictions;
        res.statusCode = 200;
        res.json(output);
      };
      await image(imagePath, async (err, imageData) => {
        try {
            const image = fs.readFileSync(imagePath);
            let tensor = tf.node.decodeImage(image);
            const resizedImage = tensor.resizeNearestNeighbor([150, 150]);
            const batchedImage = resizedImage.expandDims(0);
            const input = batchedImage.toFloat().div(tf.scalar(255));
            await loadModel(input);
            // delete image file
            fs.unlinkSync(imagePath, (error) => {
            if (error) {
                console.error(error);
            }
            });
        } catch (error) {
            res.status(500).json({message: "Internal Server Error!"});   
        }
      });
    } catch (error) {
      console.log(error)
    }
  };