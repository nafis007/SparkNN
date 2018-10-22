package com.mycompany.singlespark;

import java.io.PrintWriter;
import java.util.Map;

import backtype.storm.task.OutputCollector;
import backtype.storm.task.TopologyContext;
import backtype.storm.topology.BasicOutputCollector;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.base.BaseRichBolt;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Tuple;
import backtype.storm.tuple.Values;
import twitter4j.Status;

public class extractStatusBolt extends BaseRichBolt {
	private PrintWriter writer;
	private OutputCollector collector;

	public void prepare(Map stormConf, TopologyContext context, OutputCollector collector) {
		this.collector = collector;
		String fileName = "output" + "-" + context.getThisTaskId() + "-" + context.getThisComponentId() + ".txt";
		try {
			this.writer = new PrintWriter(stormConf.get("dirToWrite").toString() + fileName, "UTF-8");
		} catch (Exception e) {
		}
	}

	public void cleanup() {
		writer.close();
	}

	public void execute(Tuple input) {

		Status status = (Status) input.getValueByField("tweet");
		String tweetText = status.getText();
		writer.println(tweetText);

		collector.emit(new Values(tweetText));

	}

	public void declareOutputFields(OutputFieldsDeclarer declarer) {

		declarer.declare(new Fields("tweet"));
	}

}
