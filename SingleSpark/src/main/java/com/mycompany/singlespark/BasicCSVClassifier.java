package com.mycompany.singlespark;
import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//import java.util.logging.*;


import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.opencsv.CSVReader;
import java.io.BufferedReader;
import java.io.File;
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
 * This example is intended to be a simple CSV classifier that seperates the training data
 * from the test data for the classification of animals. It would be suitable as a beginner's
 * example because not only does it load CSV data into the network, it also shows how to extract the
 * data and display the results of the classification, as well as a simple method to map the lables
 * from the testing data into the results.
 *
 * @author Clay Graham
 */
public class BasicCSVClassifier {

    //private static final Logger log = LoggerFactory.getLogger(BasicCSVClassifier.class);
    
    private static final Logger log = LogManager.getLogger();
  

    //private static Map<Integer,String> eats = readEnumCSV("eats.csv");
    //private static Map<Integer,String> sounds = readEnumCSV("sounds.csv");
    //private static Map<Integer,String> classifiers = readEnumCSV("classifiers.csv");

    public static void runSimpleNN(String inputFile){

        try {
            System.out.println("hello world");
            //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
            int labelIndex = 100;     //5 values in each row of the animals.csv CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
            int numClasses = 2;     //3 classes (types of animals) in the animals data set. Classes have integer values 0, 1 or 2
            
            
            String trainingFileName = inputFile;
            //String testingFileName = "test_most100.csv";
            
            
            
            int batchSizeTraining = getRowCount(trainingFileName); //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
                                              // number of rows of the training csv file
            
            DataSet trainingData1 = readCSVDataset(trainingFileName,
                    batchSizeTraining, labelIndex, numClasses);
            
            SplitTestAndTrain testAndTrain = trainingData1.splitTestAndTrain(0.65);  //Use 65% of data for training

            // this is the data we want to classify
            //int batchSizeTest = getRowCount(testingFileName);          // number of rows of the test csv file
            
            //DataSet testData = readCSVDataset(testingFileName,
            //        batchSizeTest, labelIndex, numClasses);

            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();
            
            // make the data model for records prior to normalization, because it
            // changes the data.
            //Map<Integer,Map<String,Object>> animals = makeAnimalsForTesting(testData);

            
            List<RecordMetaData> testMetaData = testData.getExampleMetaData(RecordMetaData.class); // needed to call in the eval method
            
            
            
            //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(trainingData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
            normalizer.transform(trainingData);     //Apply normalization to the training data
            normalizer.transform(testData);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

            final int numInputs = 100; // features
            int outputNum = 2; // class labels
            int epochs = 50;
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

            //run the model
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(100));

            for( int i=0; i<epochs; i++ ) {
                model.fit(trainingData);
            }

            //evaluate the model on the test set
            //System.out.println("evaluating 1 ?");
            Evaluation eval = new Evaluation(2);
            //System.out.println("evaluating 1 ?");
            INDArray output = model.output(testData.getFeatureMatrix());
            //System.out.println("evaluating 1 ?");

            eval.eval(testData.getLabels(), output, testMetaData);     //Note we are passing in the test set metadata here
            log.info(eval.stats());                                    // for getting the predicted labels
            writeToFile("outputDL4JSimpleNN.txt",eval.stats().toString()+"\n\n");
            
            // test print
            //////////////////////////////////////getting predicted labels/////////////////////////////
            //System.out.println("test print");
            //List<Prediction> pList = eval.getPredictionsByActualClass(4);       //All predictions for actual class 2
            
            ArrayList<List<Prediction>> predictionLists = new ArrayList<List<Prediction>>();
            
            for(int i = 0; i<numClasses; i++) {
                
                List<Prediction> tempList = eval.getPredictionsByActualClass(i);
                //List<Prediction> tempList = eval.getPredictionByPredictedClass(i);
                if(tempList.size() != 0) {
                    //for(int j = 0; j< tempList.size(); j++) {
                    //    System.out.println("test pred: "+ tempList.get(j).getRecordMetaData());
                    //}
                    //System.out.println("test "+i+" "+tempList.size());
                    predictionLists.add(tempList);
                }  
            }
            //predictionLists.add(eval.getPredictionsByActualClass(10));
            
            //ArrayList<Prediction> pList = new ArrayList<Prediction>(eval.getPredictionsByActualClass(10));
            
            System.out.println("\n+++++Predictions+++++");
            int count = 1;
            for(List<Prediction> pList : predictionLists) {
                for(Prediction p : pList){
                    System.out.println("Predicted class: " + p.getPredictedClass() + ", Actual class: " + p.getActualClass());
                    //p.getRecordMetaData();
                    String metaData = p.getRecordMetaData().toString();
                    String lineNumber = metaData.substring(30, metaData.indexOf(','));
                    
                //    String out = lineNumber+","+p.getPredictedClass()+"\n";
                //    writeToFile("ouputDL4J.txt",out);
                //    count++;
                }
            }
            //////////////////////////////////////getting predicted labels/////////////////////////////
            /*List<Prediction> predictionErrors = eval.getPredictionErrors();
            System.out.println("\n\n+++++ Predictions+++++");
            for(Prediction p : predictionErrors){
                System.out.println("Predicted class: " + p.getPredictedClass() + ", Actual class: " + p.getActualClass());
            }*/
            
            

            //setFittedClassifiers(output, animals);
            //logAnimals(animals);

        } catch (Exception e){
            e.printStackTrace();
        }

    }



    public static void logAnimals(Map<Integer,Map<String,Object>> animals){
        for(Map<String,Object> a:animals.values()) {
            log.info(a.toString());
            //System.out.println("test"+a.toString());
        }
    }

    public static void setFittedClassifiers(INDArray output, Map<Integer,Map<String,Object>> animals){
        for (int i = 0; i < output.rows() ; i++) {

            // set the classification from the fitted results
 //           animals.get(i).put("classifier",
 //                   classifiers.get(maxIndex(getFloatArrayFromSlice(output.slice(i)))));

        }

    }


    /**
     * This method is to show how to convert the INDArray to a float array. This is to
     * provide some more examples on how to convert INDArray to types that are more java
     * centric.
     *
     * @param rowSlice
     * @return
     */
    public static float[] getFloatArrayFromSlice(INDArray rowSlice){
        float[] result = new float[rowSlice.columns()];
        for (int i = 0; i < rowSlice.columns(); i++) {
            result[i] = rowSlice.getFloat(i);
        }
        return result;
    }

    /**
     * find the maximum item index. This is used when the data is fitted and we
     * want to determine which class to assign the test row to
     *
     * @param vals
     * @return
     */
    public static int maxIndex(float[] vals){
        int maxIndex = 0;
        for (int i = 1; i < vals.length; i++){
            float newnumber = vals[i];
            if ((newnumber > vals[maxIndex])){
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    /**
     * take the dataset loaded for the matric and make the record model out of it so
     * we can correlate the fitted classifier to the record.
     *
     * @param testData
     * @return
     */
    public static Map<Integer,Map<String,Object>> makeAnimalsForTesting(DataSet testData){
        Map<Integer,Map<String,Object>> animals = new HashMap<>();

        INDArray features = testData.getFeatureMatrix();
        for (int i = 0; i < features.rows() ; i++) {
            INDArray slice = features.slice(i);
            Map<String,Object> animal = new HashMap();

            //set the attributes
            animal.put("yearsLived", slice.getInt(0));
           // animal.put("eats", eats.get(slice.getInt(1)));
            //animal.put("sounds", sounds.get(slice.getInt(2)));
            animal.put("weight", slice.getFloat(3));

            animals.put(i,animal);
        }
        return animals;

    }


    /*public static Map<Integer,String> readEnumCSV(String csvFileClasspath) {
        try{
            List<String> lines = IOUtils.readLines(new ClassPathResource(csvFileClasspath).getInputStream());
            Map<Integer,String> enums = new HashMap<>();
            for(String line:lines){
                String[] parts = line.split(",");
                enums.put(Integer.parseInt(parts[0]),parts[1]);
            }
            return enums;
        } catch (Exception e){
            e.printStackTrace();
            return null;
        }

    }*/
    
    
    public static Map<Integer,String> readEnumCSV(String csvFileClasspath) {
        
        CSVReader reader = null;
        List<String> lines;
        try {
            reader = new CSVReader(new FileReader(csvFileClasspath));
            Map<Integer,String> enums = new HashMap<>();
            String[] line;
            while ((line = reader.readNext()) != null) {
                enums.put(Integer.parseInt(line[0]),line[1]);
            }
            return enums;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
    
    
    

    /**
     * used for testing and training
     *
     * @param csvFileClasspath
     * @param batchSize
     * @param labelIndex
     * @param numClasses
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
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
