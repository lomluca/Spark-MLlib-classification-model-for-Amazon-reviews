package it.polito.bigdata.spark.example;

import org.apache.spark.api.java.*;
import org.apache.spark.SparkConf;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.SQLContext;

import scala.Tuple2;

import org.apache.spark.sql.DataFrame;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;


import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;

public class SparkDriver {
	
	public static void main(String[] args) {

		String inputPath;

		inputPath=args[0];
	
		// Create a configuration object and set the name of the application
		SparkConf conf=new SparkConf().setAppName("Spark Lab8");
		
		// Create a Spark Context object
		JavaSparkContext sc = new JavaSparkContext(conf);

		// Create a Spark SQL Context object
		SQLContext sqlContext = new org.apache.spark.sql.SQLContext(sc);
		
		
        	//EX 1: READ AND FILTER THE DATASET AND STORE IT INTO A DATAFRAME
		
		// To avoid parsing the comma escaped within quotes, you can use the following regex:
		// line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
		// instead of the simpler
		// line.split(",");
		// this will ignore the commas followed by an odd number of quotes.
		
		JavaRDD<String> input = sc.textFile(inputPath);
		JavaRDD<String> inputFiltered = input.filter(new Function<String, Boolean>() {

			@Override
			public Boolean call(String line) throws Exception {
				// TODO Auto-generated method stub
				String[] fields = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
				if(line.startsWith("Id") == false) {
					if(Integer.parseInt(fields[5]) > 0) {
						return true;
					}
				};
				return false;
			}
		}).cache();
		
		JavaPairRDD<String, String> linesForUser = inputFiltered.mapToPair(new PairFunction<String, String, String>() {

			@Override
			public Tuple2<String, String> call(String arg0) throws Exception {
				// TODO Auto-generated method stub
				String[] fields = arg0.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
				return new Tuple2<String, String>(fields[2], arg0);
			}
			
		});
		
		JavaPairRDD<String, Tuple2<Integer, Integer>> usersMap = inputFiltered.mapToPair(new PairFunction<String, String, Tuple2<Integer, Integer>>() {

			@Override
			public Tuple2<String, Tuple2<Integer, Integer>> call(String line) throws Exception {
				// TODO Auto-generated method stub
				Tuple2 t;
				String[] fields = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
				if(Double.parseDouble(fields[4]) / Double.parseDouble(fields[5]) > 0.9) {
					t = new Tuple2<Integer, Integer>(1, 1);
				} else {
					t = new Tuple2<Integer, Integer>(1, 0);
				}
				return new Tuple2<String, Tuple2<Integer, Integer>>(fields[2], t);
			}
			
		});
		
		JavaPairRDD<String, Tuple2<Integer, Integer>> usersCount = usersMap.reduceByKey(new Function2<Tuple2<Integer, Integer>, Tuple2<Integer, Integer>, Tuple2<Integer, Integer>>() {

			@Override
			public Tuple2<Integer, Integer> call(Tuple2<Integer, Integer> arg0, Tuple2<Integer, Integer> arg1)
					throws Exception {
				// TODO Auto-generated method stub
				return new Tuple2<Integer, Integer>(arg0._1 + arg1._1, arg0._2 + arg1._2);
			}	
		});
		
		JavaPairRDD<String, Tuple2<String, Tuple2<Integer, Integer>>> inputWithNewFields = linesForUser.join(usersCount);
		
		//features:
		//lenght text
		//number of reviews of the user
		//number of useful reviews of the user
		//score

		JavaRDD<LabeledPoint> trainingRDD = inputWithNewFields.map(new Function<Tuple2<String, Tuple2<String, Tuple2<Integer, Integer>>>, LabeledPoint>() {

			@Override
			public LabeledPoint call(Tuple2<String, Tuple2<String, Tuple2<Integer, Integer>>> arg0) throws Exception {
				// TODO Auto-generated method stub
				String[] fields = arg0._2._1.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
				double[] attributesValues = new double[4];
				attributesValues[0] = fields[9].length();
				attributesValues[1] = arg0._2._2._1;
				attributesValues[2] = arg0._2._2._2;
				attributesValues[3] = Double.parseDouble(fields[6]);
				
				Vector v = Vectors.dense(attributesValues);
				
				if(Double.parseDouble(fields[4]) / Double.parseDouble(fields[5]) > 0.9) {
					return new LabeledPoint(1.0, v);
				} else {
					return new LabeledPoint(0.0, v);
				}
			}
		});
		
		DataFrame schemaReviews = sqlContext.createDataFrame(trainingRDD, LabeledPoint.class).cache();
				
				
		// Display 5 example rows.
		schemaReviews.show(5);


		// Split the data into training and test sets (30% held out for testing)
		DataFrame[] splits = schemaReviews.randomSplit(new double[]{0.7, 0.3});
		DataFrame trainingData = splits[0];
		DataFrame testData = splits[1];

		//LOGISTIC REGRESSION
		LogisticRegression lr = new LogisticRegression();
		lr.setMaxIter(10);
		lr.setRegParam(0.01);
		
		//CLASSIFICATION TREE
		StringIndexerModel labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(trainingData);
		DecisionTreeClassifier dc = new DecisionTreeClassifier();
		dc.setImpurity("gini");
		dc.setLabelCol("indexedLabel");
		IndexToString labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedValue").setLabels(labelIndexer.labels());

        //EX 2: CREATE THE PIPELINE THAT IS USED TO BUILD THE CLASSIFICATION MODEL
		Pipeline pipelineLR =  new Pipeline().setStages(new PipelineStage[] { lr });
		Pipeline pipelineDC = new Pipeline().setStages(new PipelineStage[] { labelIndexer, dc, labelConverter });

		// Train model. Use the training set 
		PipelineModel modelLR = pipelineLR.fit(trainingData);
		PipelineModel modelDC = pipelineDC.fit(trainingData);
			
				
		/*==== EVALUATION ====*/


		// Make predictions for the test set.
		DataFrame predictionsLR = modelLR.transform(testData);
		DataFrame predictionsDC = modelDC.transform(testData);

		// Select example rows to display.
		predictionsLR.show(5);
		predictionsDC.show(5);

		// Retrieve the quality metrics. 
        MulticlassMetrics metricsDC = new MulticlassMetrics(predictionsDC.select("prediction", "indexedLabel"));
        	// Use the following command if you are using logistic regression
        MulticlassMetrics metricsLR = new MulticlassMetrics(predictionsLR.select("prediction", "label"));

	    // Confusion matrix
        Matrix confusionLR = metricsLR.confusionMatrix();
        Matrix confusionDC = metricsDC.confusionMatrix();
        System.out.println("Confusion matrix LR: \n" + confusionLR);
        System.out.println("Confusion matrix DC: \n" + confusionDC);
        
        double precisionLR = metricsLR.precision();
        double precisionDC = metricsDC.precision();
		System.out.println("Precision LR = " + precisionLR);
		System.out.println("Precision DC = " + precisionDC);
            
	    // Close the Spark context
		sc.close();
	}
}
