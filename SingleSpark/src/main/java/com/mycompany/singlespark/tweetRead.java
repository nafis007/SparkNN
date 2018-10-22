package com.mycompany.singlespark;

import java.util.Map;

import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.tuple.Fields;

import backtype.storm.task.ShellBolt;
import backtype.storm.topology.IRichBolt;

public class tweetRead extends ShellBolt implements IRichBolt {
	public tweetRead() {
		super("python",
				"E:\\(3) Semester 2 July 2018\\COMP90019_Distributed Computing Project\\SparkNN\\SingleSpark\\src\\resources\\tweetRead.py");
	}
	
	public void declareOutputFields(OutputFieldsDeclarer declarer) {
		declarer.declare(new Fields("flag","tweet_status"));
	}

	public Map<String, Object> getComponentConfiguration() {
		return null;
	}
}
