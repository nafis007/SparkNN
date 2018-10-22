package com.mycompany.singlespark;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Map;
import java.util.Random;

import backtype.storm.task.OutputCollector;
import backtype.storm.task.TopologyContext;
import backtype.storm.topology.BasicOutputCollector;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.base.BaseRichBolt;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Tuple;
import backtype.storm.tuple.Values;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;


// BaseBasicBolt can also be used in extend
public class featureMap extends BaseRichBolt {
	
	private OutputCollector collector;
	private PrintWriter writer;
	private PrintWriter baseWriter;
	private String outputDirectory;
	private String fileName;

	
	public int count;

	public int test;

	private String trainFileName;
	private static String testFileName;
	private SparkNN sNN;
	private JavaSparkContext sc;
	private TrainingMaster tm;
	private SparkDl4jMultiLayer sparkNetwork;


	public void prepare(Map conf, TopologyContext context, OutputCollector collector) {

			this.test = 0;

	        this.collector = collector;
	        this.outputDirectory = conf.get("dirToWrite").toString();
	        this.fileName = "output" +".csv";
	        this.trainFileName = "top_50_new.csv";

			this.count = 0;

			//this.testFileName = null;
			//this.testFileName = "test_top_50_new.csv";

			this.sNN = new SparkNN();
			this.sc = this.sNN.configureSpark();
			this.tm = this.sNN.configureTrainingMaster(1100);
			this.sparkNetwork = this.sNN.configureLearningNetwork(this.sc,this.tm);
	        
	        try {
				this.writer = new PrintWriter(outputDirectory+fileName, "UTF-8");
		        this.baseWriter = new PrintWriter(new FileWriter(outputDirectory+trainFileName, true));
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} 	
	}
    public void cleanup() {writer.close();}


    public void execute(Tuple input){
        int flag = Integer.parseInt(String.valueOf(input.getValueByField("flag")));
        if (flag == 1) {

        	this.count++;

        	String values = String.valueOf(input.getValueByField("tweet_status"));
        	String[] row = values.split(",");
        	writer.println(row[0]+ "," + row[1] + "," + row[2] + "," + row[3] + "," + row[4] + "," + row[5] + "," + row[6] + "," + row[7] + "," + row[8] + "," + row[9] + "," + row[10] + "," + row[11] + "," + row[12] + "," + row[13] + "," + row[14] + "," + row[15] + "," + row[16] + "," + row[17] + "," + row[18] + "," + row[19] + "," + row[20] + "," + row[21] + "," + row[22] + "," + row[23] + "," + row[24] + "," + row[25] + "," + row[26] + "," + row[27] + "," + row[28] + "," + row[29] + "," + row[30] + "," + row[31] + "," + row[32] + "," + row[33] + "," + row[34] + "," + row[35] + "," + row[36] + "," + row[37] + "," + row[38] + "," + row[39] + "," + row[40] + "," + row[41] + "," + row[42] + "," + row[43] + "," + row[44] + "," + row[45] + "," + row[46] + "," + row[47] + "," + row[48] + "," + row[49] + "," + row[50] + "," + row[51] + "," + row[52] + "," + row[53] + "," + row[54] + "," + row[55] + "," + row[56] + "," + row[57] + "," + row[58] + "," + row[59] + "," + row[60] + "," + row[61] + "," + row[62] + "," + row[63] + "," + row[64] + "," + row[65] + "," + row[66] + "," + row[67] + "," + row[68] + "," + row[69] + "," + row[70] + "," + row[71] + "," + row[72] + "," + row[73] + "," + row[74] + "," + row[75] + "," + row[76] + "," + row[77] + "," + row[78] + "," + row[79] + "," + row[80] + "," + row[81] + "," + row[82] + "," + row[83] + "," + row[84] + "," + row[85] + "," + row[86] + "," + row[87] + "," + row[88] + "," + row[89] + "," + row[90] + "," + row[91] + "," + row[92] + "," + row[93] + "," + row[94] + "," + row[95] + "," + row[96] + "," + row[97] + "," + row[98] + "," + row[99] + "," + row[100]);

			//baseWriter.close();
			//writer.close();

			if (this.test == 0) {
				baseWriter.println(row[0]+ "," + row[1] + "," + row[2] + "," + row[3] + "," + row[4] + "," + row[5] + "," + row[6] + "," + row[7] + "," + row[8] + "," + row[9] + "," + row[10] + "," + row[11] + "," + row[12] + "," + row[13] + "," + row[14] + "," + row[15] + "," + row[16] + "," + row[17] + "," + row[18] + "," + row[19] + "," + row[20] + "," + row[21] + "," + row[22] + "," + row[23] + "," + row[24] + "," + row[25] + "," + row[26] + "," + row[27] + "," + row[28] + "," + row[29] + "," + row[30] + "," + row[31] + "," + row[32] + "," + row[33] + "," + row[34] + "," + row[35] + "," + row[36] + "," + row[37] + "," + row[38] + "," + row[39] + "," + row[40] + "," + row[41] + "," + row[42] + "," + row[43] + "," + row[44] + "," + row[45] + "," + row[46] + "," + row[47] + "," + row[48] + "," + row[49] + "," + row[50] + "," + row[51] + "," + row[52] + "," + row[53] + "," + row[54] + "," + row[55] + "," + row[56] + "," + row[57] + "," + row[58] + "," + row[59] + "," + row[60] + "," + row[61] + "," + row[62] + "," + row[63] + "," + row[64] + "," + row[65] + "," + row[66] + "," + row[67] + "," + row[68] + "," + row[69] + "," + row[70] + "," + row[71] + "," + row[72] + "," + row[73] + "," + row[74] + "," + row[75] + "," + row[76] + "," + row[77] + "," + row[78] + "," + row[79] + "," + row[80] + "," + row[81] + "," + row[82] + "," + row[83] + "," + row[84] + "," + row[85] + "," + row[86] + "," + row[87] + "," + row[88] + "," + row[89] + "," + row[90] + "," + row[91] + "," + row[92] + "," + row[93] + "," + row[94] + "," + row[95] + "," + row[96] + "," + row[97] + "," + row[98] + "," + row[99] + "," + 1);
				this.testFileName = null;
			}
			else if(this.test == 1) {
				this.testFileName = this.fileName;
			}

			//baseWriter.println(row[0]+ "," + row[1] + "," + row[2] + "," + row[3] + "," + row[4] + "," + row[5] + "," + row[6] + "," + row[7] + "," + row[8] + "," + row[9] + "," + row[10] + "," + row[11] + "," + row[12] + "," + row[13] + "," + row[14] + "," + row[15] + "," + row[16] + "," + row[17] + "," + row[18] + "," + row[19] + "," + row[20] + "," + row[21] + "," + row[22] + "," + row[23] + "," + row[24] + "," + row[25] + "," + row[26] + "," + row[27] + "," + row[28] + "," + row[29] + "," + row[30] + "," + row[31] + "," + row[32] + "," + row[33] + "," + row[34] + "," + row[35] + "," + row[36] + "," + row[37] + "," + row[38] + "," + row[39] + "," + row[40] + "," + row[41] + "," + row[42] + "," + row[43] + "," + row[44] + "," + row[45] + "," + row[46] + "," + row[47] + "," + row[48] + "," + row[49] + "," + row[50] + "," + row[51] + "," + row[52] + "," + row[53] + "," + row[54] + "," + row[55] + "," + row[56] + "," + row[57] + "," + row[58] + "," + row[59] + "," + row[60] + "," + row[61] + "," + row[62] + "," + row[63] + "," + row[64] + "," + row[65] + "," + row[66] + "," + row[67] + "," + row[68] + "," + row[69] + "," + row[70] + "," + row[71] + "," + row[72] + "," + row[73] + "," + row[74] + "," + row[75] + "," + row[76] + "," + row[77] + "," + row[78] + "," + row[79] + "," + row[80] + "," + row[81] + "," + row[82] + "," + row[83] + "," + row[84] + "," + row[85] + "," + row[86] + "," + row[87] + "," + row[88] + "," + row[89] + "," + row[90] + "," + row[91] + "," + row[92] + "," + row[93] + "," + row[94] + "," + row[95] + "," + row[96] + "," + row[97] + "," + row[98] + "," + row[99] + "," + 1);

        	//
			if(this.count % 5 == 0) {
				writer.close();
				baseWriter.close();
				try {
					//BasicCSVClassifier bCC = new BasicCSVClassifier();
					//bCC.runSimpleNN(this.fileName);

					TrainingModel preBuiltModel = this.sNN.buildTrainedModel(this.trainFileName,this.testFileName,5,this.sc,this.tm, this.sparkNetwork);

					//	TrainingModel testModel = this.sNN.getTestData(this.fileName, this.testFileName);

					Evaluation evaluation = this.sNN.evaluateModel(preBuiltModel.sparkNetwork,preBuiltModel.testDataSet,preBuiltModel.testMetaData);

					//	Evaluation evaluation = this.sNN.evaluateModel(preBuiltModel.sparkNetwork,testModel.testDataSet,testModel.testMetaData);
					writeToFile("testResultGenerate.txt",evaluation.accuracy()+","+evaluation.recall()+","+evaluation.precision()+"\n");

				} catch (Exception e) {
					e.printStackTrace();

				}
				try {
					this.writer = new PrintWriter(new FileWriter(outputDirectory+fileName, true));
					this.baseWriter = new PrintWriter(new FileWriter(outputDirectory+trainFileName, true));
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}



        	

        }

        
  			
				
       	
        
    }

    public void declareOutputFields(OutputFieldsDeclarer declarer) {
    	declarer.declare(new Fields("flag","tweet_status"));
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