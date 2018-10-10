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
    private boolean useSparkLocal = true; //////////////// True for local testing

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 8;   //How many examples should be used per worker (executor) when fitting?

    @Parameter(names = "-numEpochs", description = "Number of epochs for training")
    private int numEpochs = 1;

    public static void main(String[] args) throws Exception {
        new SparkNN().entryPoint(args);
    }

    protected void entryPoint(String[] args) throws Exception {
        //Handle command line arguments
        JCommander jcmdr = new JCommander(this);
        try {
            jcmdr.parse(args);
        } catch (ParameterException e) {
            //User provides invalid input -> print the usage info
            jcmdr.usage();
            try {
                Thread.sleep(500);
            } catch (Exception e2) {
            }
            throw e;
        }

        try {
            System.out.println("hello world");
            //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
            int labelIndex = 100;     //5 values in each row of the animals.csv CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
            int numClasses = 2;     //3 classes (types of animals) in the animals data set. Classes have integer values 0, 1 or 2


            String trainingFileName = "most_100_new.csv";
            //String testingFileName = "dev_most100.csv";



            int batchSizeTraining = getRowCount(trainingFileName); //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
            // number of rows of the training csv file


            DataSet trainDataSet1 = readCSVDataset(trainingFileName,
                    batchSizeTraining, labelIndex, numClasses);

            SplitTestAndTrain testAndTrain = trainDataSet1.splitTestAndTrain(0.65);  //Use 65% of data for training

            // this is the data we want to classify
            //int batchSizeTest = getRowCount(testingFileName);          // number of rows of the test csv file

            //DataSet testDataSet = readCSVDataset(testingFileName,
            //        batchSizeTest, labelIndex, numClasses);



            DataSet trainDataSet = testAndTrain.getTrain();
            DataSet testDataSet = testAndTrain.getTest();

            // make the data model for records prior to normalization, because it
            // changes the data.
            //Map<Integer,Map<String,Object>> animals = makeAnimalsForTesting(testData);


            List<RecordMetaData> testMetaData = testDataSet.getExampleMetaData(RecordMetaData.class); // needed to call in the eval method



            //We need to normalize our data. We'll use NormalizeStandardize (which gives us mean 0, unit variance):
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(trainDataSet);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
            normalizer.transform(trainDataSet);     //Apply normalization to the training data
            normalizer.transform(testDataSet);         //Apply normalization to the test data. This is using statistics calculated from the *training* set

            final int numInputs = 100; // features
            int outputNum = 2; // class labels
            int epochs = 1000;
            long seed = 6;

            log.info("Build model....");

            /*MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(12345)
                    .l2(0.001)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new RmsProp(0.1))
                    .list()
                    .layer(0, new LSTM.Builder().nIn(numInputs).nOut(11).activation(Activation.TANH).build())
                    .layer(1, new LSTM.Builder().nIn(11).nOut(11).activation(Activation.TANH).build())
                    .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                            .nIn(11).nOut(outputNum).build())
                    .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(50).tBPTTBackwardLength(50)
                    .pretrain(false).backprop(true)
                    .build(); */



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


            //-------------------------------------------------------------
            //Set up the Spark-specific configuration
            /* How frequently should we average parameters (in number of minibatches)?
            Averaging too frequently can be slow (synchronization + serialization costs) whereas too infrequently can result
            learning difficulties (i.e., network may not converge) */
            int averagingFrequency = 10;

            //Set up Spark configuration and context
            SparkConf sparkConf = new SparkConf();
            if (useSparkLocal) {
                sparkConf.setMaster("local[*]");
            }

            ////////////////////change happening///////////////////
            // .setMaster("spark://172.1.1.1:7077").set("spark.executor.memory","1g")
            //.setMaster("spark://master:7077").set("spark.executor.memory","1g");
            sparkConf.setAppName("LSTM Character Example");

            JavaSparkContext sc = new JavaSparkContext(sparkConf);


            List<DataSet> trainDataList = trainDataSet.asList();
            List<DataSet> testDataList = testDataSet.asList();

            /*for(int i = 0; i< 100; i++){
                System.out.println(trainDataList.get(i));
            }*/

            System.out.println("Check Size: "+trainDataList.size()+" "+testDataList.size());

            JavaRDD<DataSet> trainData = sc.parallelize(trainDataList);
            JavaRDD<DataSet> testData = sc.parallelize(testDataList);

            //Set up the TrainingMaster. The TrainingMaster controls how learning is actually executed on Spark
            //Here, we are using standard parameter averaging
            //For details on these configuration options, see: https://deeplearning4j.org/spark#configuring

            /*int examplesPerDataSetObject = 1;
            ParameterAveragingTrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
                    .workerPrefetchNumBatches(2)    //Asynchronously prefetch up to 2 batches
                    .averagingFrequency(averagingFrequency)
                    .batchSizePerWorker(batchSizePerWorker)
                    .build(); */
            //Configuration for Spark training: see http://deeplearning4j.org/spark for explanation of these configuration options
            TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(batchSizePerWorker)    //Each DataSet object: contains (by default) 32 examples
                    .averagingFrequency(5)
                    .workerPrefetchNumBatches(2)            //Async prefetching: 2 examples per worker
                    .batchSizePerWorker(batchSizePerWorker)
                    .build();


            SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, conf, tm);
            //sparkNetwork.setListeners(new ScoreIterationListener(1));

            //run the model
            //MultiLayerNetwork model = new MultiLayerNetwork(conf);
            //model.init();
            //model.setListeners(new ScoreIterationListener(100));




            //Do training, and then generate and print samples from network
            for (int i = 0; i < numEpochs; i++) {
                //Perform one epoch of training. At the end of each epoch, we are returned a copy of the trained network
                sparkNetwork.fit(trainData);
                log.info("Completed Epoch {}", i);
            }

            //Perform evaluation (distributed)
            //Evaluation evaluation = sparkNetwork.evaluate(testData);
            //Evaluation evaluation = sparkNetwork.doEvaluation(testData, 64, new Evaluation(11))[0]; //Work-around for 0.9.1 bug: see https://deeplearning4j.org/releasenotes
            log.info("***** Evaluation *****");
            //log.info(evaluation.stats());



            //evaluate the model on the test set
            //System.out.println("evaluating 1 ?");
            Evaluation eval = new Evaluation(outputNum);
            //System.out.println("evaluating 1 ?");
            INDArray output = sparkNetwork.getNetwork().output(testDataSet.getFeatureMatrix());
            //System.out.println("evaluating 1 ?");

            eval.eval(testDataSet.getLabels(), output, testMetaData);     //Note we are passing in the test set metadata here
            log.info(eval.stats());                                    // for getting the predicted labels


            // test print
            //////////////////////////////////////getting predicted labels/////////////////////////////
            System.out.println("test print");
            //List<Prediction> pList = eval.getPredictionsByActualClass(4);       //All predictions for actual class 2

            ArrayList<List<Prediction>> predictionLists = new ArrayList<List<Prediction>>();

            for(int i = 0; i<numClasses; i++) {

                List<Prediction> tempList = eval.getPredictionsByActualClass(i);
                //List<Prediction> tempList = eval.getPredictionByPredictedClass(i);
                if(tempList.size() != 0) {
                    //for(int j = 0; j< tempList.size(); j++) {
                    //    System.out.println("test pred: "+ tempList.get(j).getRecordMetaData());
                    //}
                    System.out.println("test "+i+" "+tempList.size());
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
                    //String metaData = p.getRecordMetaData().toString();
                    //String lineNumber = metaData.substring(30, metaData.indexOf(','));

                    //String out = lineNumber+","+p.getPredictedClass()+"\n";
                    //writeToFile("ouputDL4J.txt",out);
                    //count++;
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
            tm.deleteTempFiles(sc);

            log.info("\n\nExample complete");

        } catch (Exception e){
            e.printStackTrace();
        }










       /* Random rng = new Random(12345);
        int lstmLayerSize = 200;                    //Number of units in each LSTM layer
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int nSamplesToGenerate = 4;                    //Number of samples to generate after each training epoch
        int nCharactersToSample = 300;                //Length of each sample to generate
        String generationInitialization = null;        //Optional character initialization; a random character is used if null
        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
        // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default

        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(new RmsProp(0.1))
            .list()
            .layer(0, new LSTM.Builder().nIn(CHAR_TO_INT.size()).nOut(lstmLayerSize).activation(Activation.TANH).build())
            .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize).activation(Activation.TANH).build())
            .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                .nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .pretrain(false).backprop(true)
            .build();




        JavaRDD<DataSet> trainingData = getTrainingData(sc);






        //Delete the temp training files, now that we are done with them
        tm.deleteTempFiles(sc);

        log.info("\n\nExample complete"); */
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

        //List<DataSet> dataList;

        //while (iterator.hasNext()) {
        //    dataList.add(iterator.next().);
        //}
        //dataList = iterator.next().asList();
        //System.out.println("get data size: "+dataList.size());
        return iterator.next();
        //return dataList;
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

    /**
     * Get the training data - a JavaRDD<DataSet>
     * Note that this approach for getting training data is a special case for this example (modelling characters), and
     * should  not be taken as best practice for loading data (like CSV etc) in general.
     */
    /*public static JavaRDD<DataSet> getTrainingData(JavaSparkContext sc) throws IOException {
        //Get data. For the sake of this example, we are doing the following operations:
        // File -> String -> List<String> (split into length "sequenceLength" characters) -> JavaRDD<String> -> JavaRDD<DataSet>
        List<String> list = getShakespeareAsList(exampleLength);
        JavaRDD<String> rawStrings = sc.parallelize(list);
        Broadcast<Map<Character, Integer>> bcCharToInt = sc.broadcast(CHAR_TO_INT);
        return rawStrings.map(new StringToDataSetFn(bcCharToInt));
    }


    private static class StringToDataSetFn implements Function<String, DataSet> {
        private final Broadcast<Map<Character, Integer>> ctiBroadcast;

        private StringToDataSetFn(Broadcast<Map<Character, Integer>> characterIntegerMap) {
            this.ctiBroadcast = characterIntegerMap;
        }

            public DataSet call(String s) throws Exception {
            //Here: take a String, and map the characters to a one-hot representation
            Map<Character, Integer> cti = ctiBroadcast.getValue();
            int length = s.length();
            INDArray features = Nd4j.zeros(1, N_CHARS, length - 1);
            INDArray labels = Nd4j.zeros(1, N_CHARS, length - 1);
            char[] chars = s.toCharArray();
            int[] f = new int[3];
            int[] l = new int[3];
            for (int i = 0; i < chars.length - 2; i++) {
                f[1] = cti.get(chars[i]);
                f[2] = i;
                l[1] = cti.get(chars[i + 1]);   //Predict the next character given past and current characters
                l[2] = i;

                features.putScalar(f, 1.0);
                labels.putScalar(l, 1.0);
            }
            return new DataSet(features, labels);
        }
    }

    //This function downloads (if necessary), loads and splits the raw text data into "sequenceLength" strings
    private static List<String> getShakespeareAsList(int sequenceLength) throws IOException {
        //The Complete Works of William Shakespeare
        //5.3MB file in UTF-8 Encoding, ~5.4 million characters
        //https://www.gutenberg.org/ebooks/100
        String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
        String tempDir = System.getProperty("java.io.tmpdir");
        String fileLocation = tempDir + "/Shakespeare.txt";    //Storage location from downloaded file
        File f = new File(fileLocation);
        if (!f.exists()) {
            FileUtils.copyURLToFile(new URL(url), f);
            System.out.println("File downloaded to " + f.getAbsolutePath());
        } else {
            System.out.println("Using existing text file at " + f.getAbsolutePath());
        }

        if (!f.exists()) throw new IOException("File does not exist: " + fileLocation);    //Download problem?

        String allData = getDataAsString(fileLocation);

        List<String> list = new ArrayList<String>();
        int length = allData.length();
        int currIdx = 0;
        while (currIdx + sequenceLength < length) {
            int end = currIdx + sequenceLength;
            String substr = allData.substring(currIdx, end);
            currIdx = end;
            list.add(substr);
        }
        return list;
    }  */

    /**
     * Load data from a file, and remove any invalid characters.
     * Data is returned as a single large String
     */
    /* private static String getDataAsString(String filePath) throws IOException {
        List<String> lines = Files.readAllLines(new File(filePath).toPath(), Charset.defaultCharset());
        StringBuilder sb = new StringBuilder();
        for (String line : lines) {
            char[] chars = line.toCharArray();
            for (int i = 0; i < chars.length; i++) {
                if (CHAR_TO_INT.containsKey(chars[i])) sb.append(chars[i]);
            }
            sb.append("\n");
        }

        return sb.toString();
    }

    /**
     * Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
     * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
     * Note that the initalization is used for all samples
     *
     * @param initialization     String, may be null. If null, select a random character as initialization for all samples
     * @param charactersToSample Number of characters to sample from network (excluding initialization)
     * @param net                MultiLayerNetwork with one or more LSTM/RNN layers and a softmax output layer
     */

    /* private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net, Random rng,
                                                        Map<Integer, Character> intToChar, int charactersToSample, int numSamples) {
        //Set up initialization. If no initialization: use a random character
        if (initialization == null) {
            int randomCharIdx = rng.nextInt(intToChar.size());
            initialization = String.valueOf(intToChar.get(randomCharIdx));
        }

        //Create input for initialization
        INDArray initializationInput = Nd4j.zeros(numSamples, intToChar.size(), initialization.length());
        char[] init = initialization.toCharArray();
        for (int i = 0; i < init.length; i++) {
            int idx = CHAR_TO_INT.get(init[i]);
            for (int j = 0; j < numSamples; j++) {
                initializationInput.putScalar(new int[]{j, idx, i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for (int i = 0; i < numSamples; i++) sb[i] = new StringBuilder(initialization);

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension(output.size(2) - 1, 1, 0);    //Gets the last time step output

        for (int i = 0; i < charactersToSample; i++) {
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(numSamples, intToChar.size());
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for (int s = 0; s < numSamples; s++) {
                double[] outputProbDistribution = new double[intToChar.size()];
                for (int j = 0; j < outputProbDistribution.length; j++)
                    outputProbDistribution[j] = output.getDouble(s, j);
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng);

                nextInput.putScalar(new int[]{s, sampledCharacterIdx}, 1.0f);        //Prepare next time step input
                sb[s].append(intToChar.get(sampledCharacterIdx));    //Add sampled character to StringBuilder (human readable output)
            }

            output = net.rnnTimeStep(nextInput);    //Do one time step of forward pass
        }

        String[] out = new String[numSamples];
        for (int i = 0; i < numSamples; i++) out[i] = sb[i].toString();
        return out;
    }
    */

    /**
     * Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     *
     * @param distribution Probability distribution over classes. Must sum to 1.0
     */
    /*private static int sampleFromDistribution(double[] distribution, Random rng) {
        double d = rng.nextDouble();
        double sum = 0.0;
        for (int i = 0; i < distribution.length; i++) {
            sum += distribution[i];
            if (d <= sum) return i;
        }
        //Should never happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum);
    }
    */
    /**
     * A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc
     */
    /* private static char[] getValidCharacters() {
        List<Character> validChars = new LinkedList<>();
        for (char c = 'a'; c <= 'z'; c++) validChars.add(c);
        for (char c = 'A'; c <= 'Z'; c++) validChars.add(c);
        for (char c = '0'; c <= '9'; c++) validChars.add(c);
        char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
        for (char c : temp) validChars.add(c);
        char[] out = new char[validChars.size()];
        int i = 0;
        for (Character c : validChars) out[i++] = c;
        return out;
    }

    public static Map<Integer, Character> getIntToChar() {
        Map<Integer, Character> map = new HashMap<>();
        char[] chars = getValidCharacters();
        for (int i = 0; i < chars.length; i++) {
            map.put(i, chars[i]);
        }
        return map;
    }

    public static Map<Character, Integer> getCharToInt() {
        Map<Character, Integer> map = new HashMap<>();
        char[] chars = getValidCharacters();
        for (int i = 0; i < chars.length; i++) {
            map.put(chars[i], i);
        }
        return map;
    }*/
}

