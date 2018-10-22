package com.mycompany.singlespark;

import backtype.storm.Config;
import backtype.storm.LocalCluster;
import backtype.storm.topology.TopologyBuilder;


public class TopologyMain {
	 public static void main(String[] args) throws InterruptedException {
		 
		 //Topology definition
		 TopologyBuilder builder = new TopologyBuilder();
		 builder.setSpout("tweets-collector", new twitterSpout());
		 builder.setBolt("text-extractor", new extractStatusBolt()).shuffleGrouping("tweets-collector"); 
		 builder.setBolt("tweet-extractor", new tweetRead()).shuffleGrouping("text-extractor"); 
		 builder.setBolt("feature-extractor", new featureMap()).shuffleGrouping("tweet-extractor"); 
		 //Configuration
		 Config conf = new Config();
		 conf.setDebug(true);
		 conf.put("dirToWrite", "E:\\(3) Semester 2 July 2018\\COMP90019_Distributed Computing Project\\SparkNN\\SingleSpark\\");
    
    
		 LocalCluster cluster = new LocalCluster();
		 cluster.submitTopology("twitter-direct", conf, builder.createTopology());
		 Thread.sleep(12000000);
		 cluster.shutdown();
	 }
}
