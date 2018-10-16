/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.mycompany.singlespark;

/**
 *
 * @author NAFIS
 */
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
//import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;


import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//import java.util.logging.*;


import java.util.List;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.datavec.api.records.metadata.RecordMetaData;
import org.deeplearning4j.eval.meta.Prediction;
import org.nd4j.linalg.dataset.SplitTestAndTrain;



/**
 * LSTM + Spark character modelling example
 * Example: Train a LSTM RNN to generates text, one character at a time.
 * Training here is done on Spark
 *
 * See dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/character/LSTMCharModellingExample.java
 * for the single-machine version of this example
 *
 * To run the example locally: Run the example as-is. The example is set up to use Spark local by default.
 * NOTE: Spark local should only be used for development/testing. For data parallel training on a single machine
 * (for example, multi-GPU systems) instead use ParallelWrapper (which is faster than using Spark for training on a single machine).
 * See for example MultiGpuLenetMnistExample in dl4j-cuda-specific-examples
 *
 * To run the example using Spark submit (for example on a cluster): pass "-useSparkLocal false" as the application argument,
 * OR first modify the example by setting the field "useSparkLocal = false"
 *
 * @author Alex Black
 */
public class SparkNN {
    private static final Logger log = LogManager.getLogger();


    @Parameter(names = "-useSparkLocal false", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private static boolean useSparkLocal = true; //////////////// True for local testing

    //@Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    //private static int batchSizePerWorker;   //How many examples should be used per worker (executor) when fitting? was 8

    //@Parameter(names = "-numEpochs", description = "Number of epochs for training")
    //private static int numEpochs;

    private static String outputFileName = "outputDL4J.txt";

    private static String predictionFileName = "predictions.txt";

    private static String trainingFileNameGlobal = "top_50_new.csv";

    private static String testingFileNameGlobal = "test_top_50_new.csv";

    //private static String testingFileNameGlobal = null;

    private static int numInputs = 100; // features   // for us 100

    private static int outputNum = 3; // class labels // for us 2 or 3

    private static int labelIndex = numInputs;

    private static int numClasses = outputNum;

    /*public SparkNN(int numEpochs, int batchSizePerWorker) {
        this.numEpochs = numEpochs;
        this.batchSizePerWorker = batchSizePerWorker;
    }*/

    public static void main(String[] args) throws Exception {
        JavaSparkContext sc = configureSpark();

        /*for (int i = 1; i<11; i+=2){ // number of epocs
            String dataFileName = "forGraphEpoc_"+i+".txt";
            try (FileWriter fw = new FileWriter(dataFileName)) {
            } catch (IOException e) {
                e.printStackTrace();
            }
            for (int j = 100; j<=3000; j+=100) {  // batch size per worker

                long startTime = System.currentTimeMillis();
                TrainingMaster tm = configureTrainingMaster(j);
                SparkDl4jMultiLayer sparkNetwork = configureLearningNetwork(sc,tm);
                TrainingModel preBuiltModel = buildTrainedModel(trainingFileNameGlobal,testingFileNameGlobal,i,sc,tm, sparkNetwork);

                long stopTime = System.currentTimeMillis();
                long elapsedTime = stopTime - startTime;


                Evaluation evaluation = evaluateModel(preBuiltModel.sparkNetwork,preBuiltModel.testDataSet,preBuiltModel.testMetaData);

                //writeToFile("testResultGenerate.txt",evaluation.accuracy()+"\n");
                String outData = i+","+j+","+evaluation.accuracy()+","+elapsedTime+"\n";
                writeToFile(dataFileName,outData);
            }
        }*/


        TrainingMaster tm = configureTrainingMaster(1100);
        SparkDl4jMultiLayer sparkNetwork = configureLearningNetwork(sc,tm);

        TrainingModel trainingModel = buildTrainedModel(trainingFileNameGlobal,testingFileNameGlobal,5,sc,tm,sparkNetwork);
        evaluateModel(trainingModel.sparkNetwork,trainingModel.testDataSet,trainingModel.testMetaData);



    }



    public static TrainingModel buildTrainedModel(String inputFile, String testFile, int numEpochs, JavaSparkContext sc,
                                                          TrainingMaster tm, SparkDl4jMultiLayer sparkNetwork) throws IOException, InterruptedException {

        //TrainingModel trainingModel;

        //try {
        System.out.println("hello world");
        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        //int labelIndex = 100;     //5 values in each row of the animals.csv CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        //int numClasses = 2;     //3 classes (types of animals) in the animals data set. Classes have integer values 0, 1 or 2

        DataSet trainDataSet;
        DataSet testDataSet;

        if (testFile == null) {
            String trainingFileName = inputFile;


            // number of rows of the training csv file
            int batchSizeTraining = getRowCount(trainingFileName); //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)

            //////////////////// This is for split evaluation/////////////////////////////////
            DataSet wholeDataSet = readCSVDataset(trainingFileName,
                    batchSizeTraining, labelIndex, numClasses);

            SplitTestAndTrain testAndTrain = wholeDataSet.splitTestAndTrain(0.65);  //Use 65% of data for training

            trainDataSet = testAndTrain.getTrain();
            testDataSet = testAndTrain.getTest();
        }

        else {
            String trainingFileName = inputFile;
            String testingFileName = testFile;


            // number of rows of the training csv file
            int batchSizeTraining = getRowCount(trainingFileName); //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
            int batchSizeTest = getRowCount(testingFileName);          // number of rows of the test csv file


            //////////////This is for separate evaluation specially to generate output label/////////////////////
            trainDataSet = readCSVDataset(trainingFileName, batchSizeTraining, labelIndex, numClasses);
            testDataSet = readCSVDataset(testingFileName, batchSizeTest, labelIndex, numClasses);
        }




        List<RecordMetaData> testMetaData = testDataSet.getExampleMetaData(RecordMetaData.class); // needed to call in the eval method



        //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainDataSet);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(trainDataSet);     //Apply normalization to the training data
        normalizer.transform(testDataSet);         //Apply normalization to the test data. This is using statistics calculated from the *training* set



        //-------------------------------------------------------------
        //Set up the Spark-specific configuration
        /* How frequently should we average parameters (in number of minibatches)?
        Averaging too frequently can be slow (synchronization + serialization costs) whereas too infrequently can result
        learning difficulties (i.e., network may not converge) */
        int averagingFrequency = 10;



        List<DataSet> trainDataList = trainDataSet.asList();
        List<DataSet> testDataList = testDataSet.asList();


        String dataSplitSize = "Check Size: "+trainDataList.size()+" "+testDataList.size()+"\n";
        writeToFile(outputFileName,dataSplitSize);

        JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
        JavaRDD<DataSet> testData = sc.parallelize(testDataList);



        //Do training, and then generate and print samples from network
        for (int i = 0; i < numEpochs; i++) {
            //Perform one epoch of training. At the end of each epoch, we are returned a copy of the trained network
            sparkNetwork.fit(trainData);
            log.info("Completed Epoch {}", i);
        }

        //tm.deleteTempFiles(sc);

        TrainingModel trainingModel = new TrainingModel(sparkNetwork,testDataSet,testMetaData);

        log.info("\n\nExample complete");

        return trainingModel;

        //} catch (Exception e){
            /*e.printStackTrace();
            writeToFile("exception.txt",e.toString());
            return null;
        } */
    }

    public static Evaluation evaluateModel(SparkDl4jMultiLayer sparkNetwork, DataSet testDataSet, List<RecordMetaData> testMetaData) {
        log.info("***** Evaluation *****");

        //evaluate the model on the test set
        //System.out.println("evaluating 1 ?");
        Evaluation eval = new Evaluation(numClasses);
        //System.out.println("evaluating 1 ?");
        INDArray output = sparkNetwork.getNetwork().output(testDataSet.getFeatureMatrix());
        //System.out.println("evaluating 1 ?");
        eval.eval(testDataSet.getLabels(), output, testMetaData);     //Note we are passing in the test set metadata here
        log.info(eval.stats());                                    // for getting the predicted labels
        writeToFile(outputFileName,eval.stats()+"\n\n");

        // test print
        //////////////////////////////////////getting predicted labels/////////////////////////////
        //List<Prediction> pList = eval.getPredictionsByActualClass(4);       //All predictions for actual class 2

        ArrayList<List<Prediction>> predictionLists = new ArrayList<List<Prediction>>();

        for(int i = 0; i<numClasses; i++) {

            List<Prediction> tempList = eval.getPredictionsByActualClass(i);
            //List<Prediction> tempList = eval.getPredictionByPredictedClass(i);
            if(tempList.size() != 0) {
                System.out.println("test "+i+" "+tempList.size());
                predictionLists.add(tempList);
            }
        }

        System.out.println("\n+++++Predictions+++++");
        int count = 1;
        for(List<Prediction> pList : predictionLists) {
            for(Prediction p : pList){
                System.out.println("Predicted class: " + p.getPredictedClass() + ", Actual class: " + p.getActualClass());
                //p.getRecordMetaData();
                    String metaData = p.getRecordMetaData().toString();
                    String lineNumber = metaData.substring(30, metaData.indexOf(','));

                    //System.out.println(metaData);

                    String out = lineNumber+","+p.getPredictedClass()+"\n";
                    writeToFile(predictionFileName,out);
                //count++;
            }
        }
        return eval;
    }

    public static JavaSparkContext configureSpark(){
        //Set up Spark configuration and context
        try (FileWriter fw = new FileWriter(outputFileName)) {
        } catch (IOException e) {
            e.printStackTrace();
        }

        try (FileWriter fw = new FileWriter(predictionFileName)) {
        } catch (IOException e) {
            e.printStackTrace();
        }


        SparkConf sparkConf = new SparkConf();
        if (useSparkLocal) {
            sparkConf.setMaster("local[*]");
        }
        ////////////////////change happening///////////////////
        // .setMaster("spark://172.1.1.1:7077").set("spark.executor.memory","1g")
        //.setMaster("spark://master:7077").set("spark.executor.memory","1g");
        sparkConf.setAppName("Twitter Location Example");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        return sc;
    }

    public static TrainingMaster configureTrainingMaster(int batchSizePerWorker) {
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)    //Each DataSet object: contains (by default) 32 examples
                .averagingFrequency(5)
                .workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker was 2 before
                .batchSizePerWorker(batchSizePerWorker)
                .build();

        return tm;
    }

    public static SparkDl4jMultiLayer configureLearningNetwork(JavaSparkContext sc, TrainingMaster tm) {

        //final int numInputs = 100; // features
        //int outputNum = 2; // class labels
        int epochs = 1000;
        long seed = 6;

        log.info("Build model....");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(outputNum).build())
                .layer(1, new DenseLayer.Builder().nIn(outputNum).nOut(outputNum).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nIn(outputNum).nOut(outputNum).build())
                .backprop(true).pretrain(false)
                .build();


        SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, conf, tm);

        return sparkNetwork;
    }


    private static DataSet readCSVDataset(
            String csvFileClasspath, int batchSize, int labelIndex, int numClasses)
            throws IOException, InterruptedException{

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(csvFileClasspath)));

        //DataSetIterator iterator = new RecordReaderDataSetIterator(rr,batchSize,labelIndex,numClasses);

        // Above line is changed into the bottom two lines, for getting predicted labels. we need to
        // setCollectMetaData to true to collect the meta data in the necessary method calls
        RecordReaderDataSetIterator iterator = new RecordReaderDataSetIterator(rr,batchSize,labelIndex,numClasses);
        iterator.setCollectMetaData(true);

        return iterator.next();
    }


    public static int getRowCount(String fileName) throws FileNotFoundException, IOException {
        BufferedReader bufferedReader = new BufferedReader(new FileReader(fileName));
        String input;
        int count = 0;
        while((input = bufferedReader.readLine()) != null)
        {
            count++;
        }

        System.out.println("Count : "+count);

        return count;
    }

    public static void writeToFile(String FileName, String Content) {
        try
        {
            FileWriter fw = new FileWriter(FileName,true); //the true will append the new data
            fw.write(Content);//appends the string to the file
            fw.close();
        }
        catch(IOException ioe)
        {
            System.err.println("IOException: " + ioe.getMessage());
        }
    }
}

