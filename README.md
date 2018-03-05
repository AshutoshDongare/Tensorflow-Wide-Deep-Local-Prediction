# Tensorflow Local Prediction 
## Overview 

**If you do not want to deploy full Tensorflow serving for production but still want to be able to perform local predictions and save the results, this sample will help you achieve that.**

This sample exports Tensorflow estimator trained model to a local directory, loads & runs the saved model and saves the results to a CSV file. 

Find the details of [Tensorflow Wide and Deep model here.](https://www.tensorflow.org/tutorials/wide_and_deep)

 
The [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/Census+Income) contains over 48,000 samples with attributes including age, occupation, education, and income (a binary label, either `>50K` or `<=50K`). The dataset is split into roughly 32,000 training and 16,000 testing samples.

## Running the code
### Setup

**Setup the tensorflow wide & Deep as described in the orginal sample given below**
 
The [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/Census+Income) that this sample uses for training is hosted by the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/). We have provided a script that downloads and cleans the necessary files.

```
python data_download.py
```

This will download the files to `/tmp/census_data`. To change the directory, set the `--data_dir` flag.

### Training
You can run the code locally as follows:

```
python wide_deep.py
```

The model is saved to `/tmp/census_model` by default, which can be changed using the `--model_dir` flag.

To run the *wide* or *deep*-only models, set the `--model_type` flag to `wide` or `deep`. Other flags are configurable as well; see `wide_deep.py` for details.

The final accuracy should be over 83% with any of the three model types.

### TensorBoard

Run TensorBoard to inspect the details about the graph and training progression.

```
tensorboard --logdir=/tmp/census_model
```

## Exporting Trained Model

We will add code to export trained model. Add below code snippet right after the training the model and run the training and export.

```
  '''Export Trained Model for Serving'''
  wideColumns, DeepColumns = build_model_columns()
  feature_columns = DeepColumns
  feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
  export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
  servable_model_dir = "/tmp/census_exported"
  servable_model_path = model.export_savedmodel(servable_model_dir, export_input_fn)
  print("Done Exporting at Path - %s", servable_model_path )
```

Please note down the export path with the new export folder which looks something like ```1520271391```. This folder name would be different for each export.
 
## Running Predictions and Saving the results

```wide_deep_predict.py``` contains code for loading saved model to run on input data ```census_input.csv``` and save the results to output CSV file ```census_output.csv```

Configure exported model path in ```wide_deep_predict.py``` as ```exported_path = '/tmp/census_exported/1520271391'``` 

This code loads the exported model, reads input CSV line by line, prepares model input for prediction, runs prediction and writes results in Output CSV file.

Output file contains all input parameters in addition of results of prediction named as ```Predicted_income_bracket``` and ```probability```. Note that the first result column has values 0 or 1. Label value given in the training (Positive Label) will be treated as 1, ```>50K``` in this case. If required you can write actual label values to CSV by checking the results.
