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
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.util.ModelSerializer;


// import weka.classifiers.Evaluation;
// import weka.classifiers.bayes.NaiveBayes;
// import weka.core.Instances;
// import weka.core.converters.ArffSaver;
// import weka.core.converters.CSVLoader;


// BaseBasicBolt can also be used in extend
public class simpleBolt extends BaseRichBolt {
	
	private OutputCollector collector;
	private PrintWriter writer;
	private PrintWriter resultWriter; 
	private String outputDirectory;
	private String fileName;

	private  String testFileName;
	private SparkNN sNN;
	private JavaSparkContext sc;
	private TrainingMaster tm;
	private SparkDl4jMultiLayer sparkNetwork;

	// private Instances test;
	public int count;
	public void prepare(Map conf, TopologyContext context, OutputCollector collector) {
		
	        this.collector = collector;
	        this.outputDirectory = conf.get("dirToWrite").toString();
	        this.fileName = "output" +".csv";
			this.testFileName = null;

	        count = 0;
	        this.sNN = new SparkNN();
	        this.sc = this.sNN.configureSpark();
			this.tm = this.sNN.configureTrainingMaster(1100);
			this.sparkNetwork = this.sNN.configureLearningNetwork(this.sc,this.tm);
	        //this.tm = this.sNN.configureTrainingMaster();
	        //this.sparkNetwork = this.sNN.configureLearningNetwork(this.sc,this.tm);

	        try {
				this.writer = new PrintWriter(outputDirectory+fileName, "UTF-8");
				this.resultWriter = new PrintWriter(outputDirectory+"prediction"+".txt", "UTF-8");
				//For Top 50
				//writer.println("auspol,follow,ssm,australia,equal,vote,survey,thisisaustralia,springst,lnp,votey,marriag,countri,pleas,futur,right,abbott,past,ensur,win,peopl,time,artist,yes,free,day,play,love,look,marriageequ,support,join,tweet,govern,australian,thank,comment,melbourn,toni,turnbul,govt,labor,countrymus,job,polit,insid,postal,call,thx,jeffcrew,de,die,ich,und,der,en,la,das,je,que,ist,ein,nicht,auf,es,le,mit,un,loves,du,am,im,von,den,et,pas,na,des,auch,ik,een,les,berlin,se,zu,al,mal,da,te,noch,het,van,ja,bei,times,aber,il,ne,wie,days,location" );
				//For Most 100
				//writer.println("auspol,follow,ssm,de,australia,equal,vote,die,survey,ich,thisisaustralia,lnp,springst,countri,marriag,votey,und,pleas,der,love,futur,time,peopl,abbott,right,en,la,win,past,das,ensur,day,je,que,yes,ist,free,look,artist,am,ein,thank,nicht,auf,play,tweet,es,le,support,job,un,mit,re,join,marriageequ,du,im,comment,via,govern,australian,call,melbourn,toni,den,von,et,live,turnbul,pas,na,govt,mal,polit,chang,al,des,hope,insid,labor,auch,do,les,een,ik,se,oh,zu,berlin,countrymus,aus,da,postal,thx,fuck,believ,parti,start,plebiscit,tri,location");

				//test = new Instances(new BufferedReader(new FileReader(outputDirectory + "test.arff")));
		        //test.setClassIndex(test.numAttributes() - 1);
		        
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} 	
	}
    public void cleanup() {writer.close();resultWriter.close();}


    public void execute(Tuple input){
        int rowNo = input.getInteger(0);
        /* String location = "";
        if(Integer.parseInt(input.getString(101))== 1) {
        	location = "Melbounrne";
        }else {
        	location = "Not Melbounrne";
        } */

        writer.println(input.getString(1) + "," + input.getString(2) + "," + input.getString(3) + "," + input.getString(4) + "," + input.getString(5) + "," + input.getString(6) + "," + input.getString(7) + "," + input.getString(8) + "," + input.getString(9) + "," + input.getString(10) + "," + input.getString(11) + "," + input.getString(12) + "," + input.getString(13) + "," + input.getString(14) + "," + input.getString(15) + "," + input.getString(16) + "," + input.getString(17) + "," + input.getString(18) + "," + input.getString(19) + "," + input.getString(20) + "," + input.getString(21) + "," + input.getString(22) + "," + input.getString(23) + "," + input.getString(24) + "," + input.getString(25) + "," + input.getString(26) + "," + input.getString(27) + "," + input.getString(28) + "," + input.getString(29) + "," + input.getString(30) + "," + input.getString(31) + "," + input.getString(32) + "," + input.getString(33) + "," + input.getString(34) + "," + input.getString(35) + "," + input.getString(36) + "," + input.getString(37) + "," + input.getString(38) + "," + input.getString(39) + "," + input.getString(40) + "," + input.getString(41) + "," + input.getString(42) + "," + input.getString(43) + "," + input.getString(44) + "," + input.getString(45) + "," + input.getString(46) + "," + input.getString(47) + "," + input.getString(48) + "," + input.getString(49) + "," + input.getString(50) + "," + input.getString(51) + "," + input.getString(52) + "," + input.getString(53) + "," + input.getString(54) + "," + input.getString(55) + "," + input.getString(56) + "," + input.getString(57) + "," + input.getString(58) + "," + input.getString(59) + "," + input.getString(60) + "," + input.getString(61) + "," + input.getString(62) + "," + input.getString(63) + "," + input.getString(64) + "," + input.getString(65) + "," + input.getString(66) + "," + input.getString(67) + "," + input.getString(68) + "," + input.getString(69) + "," + input.getString(70) + "," + input.getString(71) + "," + input.getString(72) + "," + input.getString(73) + "," + input.getString(74) + "," + input.getString(75) + "," + input.getString(76) + "," + input.getString(77) + "," + input.getString(78) + "," + input.getString(79) + "," + input.getString(80) + "," + input.getString(81) + "," + input.getString(82) + "," + input.getString(83) + "," + input.getString(84) + "," + input.getString(85) + "," + input.getString(86) + "," + input.getString(87) + "," + input.getString(88) + "," + input.getString(89) + "," + input.getString(90) + "," + input.getString(91) + "," + input.getString(92) + "," + input.getString(93) + "," + input.getString(94) + "," + input.getString(95) + "," + input.getString(96) + "," + input.getString(97) + "," + input.getString(98) + "," + input.getString(99) + "," + input.getString(100) + "," + input.getString(101));

        if(rowNo%5000 == 0 ) {
        	writer.close();
			//SparkNN sNN = new SparkNN();

			try {
				//BasicCSVClassifier bCC = new BasicCSVClassifier();
				//bCC.runSimpleNN(this.fileName);

				TrainingModel preBuiltModel = this.sNN.buildTrainedModel(this.fileName,this.testFileName,5,this.sc,this.tm, this.sparkNetwork);


				Evaluation evaluation = this.sNN.evaluateModel(preBuiltModel.sparkNetwork,preBuiltModel.testDataSet,preBuiltModel.testMetaData);
				writeToFile("testResultGenerate.txt",evaluation.accuracy()+"\n");

			} catch (Exception e) {
				e.printStackTrace();

			}
           /* try {
            	// load CSV
                CSVLoader loader = new CSVLoader();
				loader.setSource(new File(outputDirectory+fileName));
				Instances data = loader.getDataSet();
				// save ARFF
				ArffSaver saver = new ArffSaver();
	            saver.setInstances(data);
	            saver.setFile(new File(outputDirectory + "Top_50_"+rowNo+".arff"));
	            saver.writeBatch();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
                        
            try {
            	BufferedReader breader = null;
            	breader = new BufferedReader(new FileReader(outputDirectory + "Top_50_"+rowNo+".arff"));
            	Instances train = new Instances(breader);
            	train.setClassIndex(train.numAttributes() - 1);
            	breader.close();
            	
            	
            	NaiveBayes nb = new NaiveBayes();
            	nb.buildClassifier(train);
				Evaluation eval = new Evaluation(train);
				eval.crossValidateModel(nb, train, 10, new Random(1));
				resultWriter.println("Results after building the model with " + rowNo + " Instances\n");
				resultWriter.println(eval.toSummaryString("\nResult\n========\n",true));
				resultWriter.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
				
				/*
				resultWriter.println("Prediction after building the model with " + rowNo + " Instances\n");
		        for(int i=0; i<test.numInstances(); i++) {
		            System.out.println(test.instance(i));
		            double index = nb.classifyInstance(test.instance(i));
		            String className = train.attribute(4).value((int)index);
		            resultWriter.println(className);
		        }*/
            try{
            	this.writer = new PrintWriter(new FileWriter(outputDirectory+fileName, true));
			} catch (Exception e) {
				e.printStackTrace();
			}
			
        }	
       	
        
    }

    public void declareOutputFields(OutputFieldsDeclarer declarer) {
    	//For Most 100
    	//declarer.declare(new Fields("count","auspol","follow","ssm","de","australia","equal","vote","die","survey","ich","thisisaustralia","lnp","springst","countri","marriag","votey","und","pleas","der","love","futur","time","peopl","abbott","right","en","la","win","past","das","ensur","day","je","que","yes","ist","free","look","artist","am","ein","thank","nicht","auf","play","tweet","es","le","support","job","un","mit","re","join","marriageequ","du","im","comment","via","govern","australian","call","melbourn","toni","den","von","et","live","turnbul","pas","na","govt","mal","polit","chang","al","des","hope","insid","labor","auch","do","les","een","ik","se","oh","zu","berlin","countrymus","aus","da","postal","thx","fuck","believ","parti","start","plebiscit","tri","location"));
    	//For Top 50
    	declarer.declare(new Fields("count","auspol","follow","ssm","australia","equal","vote","survey","thisisaustralia","springst","lnp","votey","marriag","countri","pleas","futur","right","abbott","past","ensur","win","peopl","time","artist","yes","free","day","play","love","look","marriageequ","support","join","tweet","govern","australian","thank","comment","melbourn","toni","turnbul","govt","labor","countrymus","job","polit","insid","postal","call","thx","jeffcrew","de","die","ich","und","der","en","la","das","je","que","ist","ein","nicht","auf","es","le","mit","un","loves","du","am","im","von","den","et","pas","na","des","auch","ik","een","les","berlin","se","zu","al","mal","da","te","noch","het","van","ja","bei","times","aber","il","ne","wie","days","location"));
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