import tensorflow as tf
import os
import numpy as np

exported_path = '/tmp/census_exported/1520271391'
predictionoutputfile = 'census_output.csv'
predictioninputfile = 'census_input.csv'


def main():
	with tf.Session() as sess:
		# load the saved model
		tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], exported_path)
		
		# get the predictor , refer tf.contrib.predictor
		predictor = tf.contrib.predictor.from_saved_model(exported_path)
		
		prediction_OutFile = open(predictionoutputfile, 'w')
		
		#Write Header for CSV file
		prediction_OutFile.write("age, workclass, fnlwgt, education, education_num,marital_status, occupation, relationship, race, gender,capital_gain, capital_loss, hours_per_week, native_country,predicted_income_bracket,probability")
		prediction_OutFile.write('\n')
		
		# Read file and create feature_dict for each record
		with open(predictioninputfile) as inf:
			# Skip header
			next(inf)
			for line in inf:

				# Read data, using python, into our features
				age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, gender, capital_gain, capital_loss, hours_per_week, native_country = line.strip().split(",")
				
				# Create a feature_dict for train.example - Get Feature Columns using
				feature_dict = {
					'age': _float_feature(value=int(age)),
					'workclass': _bytes_feature(value=workclass.encode()),
					'fnlwgt': _float_feature(value=int(fnlwgt)),
					'education': _bytes_feature(value=education.encode()),
					'education_num': _float_feature(value=int(education_num)),
					'marital_status': _bytes_feature(value=marital_status.encode()),
					'occupation': _bytes_feature(value=occupation.encode()),
					'relationship': _bytes_feature(value=relationship.encode()),
					'race': _bytes_feature(value=race.encode()),
					'gender': _bytes_feature(value=gender.encode()),
					'capital_gain': _float_feature(value=int(capital_gain)),
					'capital_loss': _float_feature(value=int(capital_loss)),
					'hours_per_week': _float_feature(value=float(hours_per_week)),
					'native_country': _bytes_feature(value=native_country.encode()),
				}
				
				# Prepare model input
				
				model_input = tf.train.Example(features=tf.train.Features(feature=feature_dict))
				
				model_input = model_input.SerializeToString()
				output_dict = predictor({"inputs": [model_input]})
				
				print(" prediction Label is ", output_dict['classes'])
				print('Probability : ' + str(output_dict['scores']))
				
				# Positive label = 1
				prediction_OutFile.write(str(age)+ "," + workclass+ "," + str(fnlwgt)+ "," + education+ "," + str(education_num) + "," + marital_status + "," + occupation + "," + relationship + "," + race+ "," +gender+ "," + str(capital_gain)+ "," + str(capital_loss)+ "," + str(hours_per_week)+ "," + native_country+ ",")
				label_index = np.argmax(output_dict['scores'])
				prediction_OutFile.write(str(label_index))
				prediction_OutFile.write(',')
				prediction_OutFile.write(str(output_dict['scores'][0][label_index]))
				prediction_OutFile.write('\n')
	
	prediction_OutFile.close()


def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == "__main__":
	main()
