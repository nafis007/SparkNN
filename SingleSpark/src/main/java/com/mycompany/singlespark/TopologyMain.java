package com.mycompany.singlespark;

import backtype.storm.Config;
import backtype.storm.LocalCluster;
import backtype.storm.topology.TopologyBuilder;

public class TopologyMain {
    public static void main(String[] args) throws Exception {

        //Topology definition
        TopologyBuilder builder = new TopologyBuilder();
        builder.setSpout("File-Reader-Spout", new fileReaderSpout());
        builder.setBolt("Simple-Bolt", new simpleBolt()).shuffleGrouping("File-Reader-Spout");

        //Configuration
        Config conf = new Config();
        conf.setDebug(true);
        conf.put("initialData", "top_50_new.csv");
        //conf.put("initialData", "E:/eclipse/incrementalLearning/most_100_new.csv");
        conf.put("dirToWrite", "E:\\(3) Semester 2 July 2018\\COMP90019_Distributed Computing Project\\SparkNN\\SingleSpark\\");

        LocalCluster cluster = new LocalCluster();
        try{
        	cluster.submitTopology("File-Reader-Topology", conf, builder.createTopology());
        	Thread.sleep(120000);
        } finally{
        	cluster.shutdown();
        }
    }
}

