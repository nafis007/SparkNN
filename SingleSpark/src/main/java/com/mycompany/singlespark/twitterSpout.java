package com.mycompany.singlespark;

import java.io.PrintWriter;
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;



import backtype.storm.spout.SpoutOutputCollector;
import backtype.storm.task.TopologyContext;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.base.BaseRichSpout;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Values;
import backtype.storm.utils.Utils;


import twitter4j.*;
import twitter4j.Status;
import twitter4j.conf.ConfigurationBuilder;
import twitter4j.json.DataObjectFactory;

public class twitterSpout extends BaseRichSpout {
	//private PrintWriter writer;
	private LinkedBlockingQueue<Status> queue;
    //stream of tweets
    private TwitterStream twitterStream;

    private SpoutOutputCollector collector;
    public void open(Map conf, TopologyContext context,SpoutOutputCollector collector) {
    	
    	
        this.collector = collector;
        ConfigurationBuilder cb = new ConfigurationBuilder();
        cb.setDebugEnabled(true)
                .setOAuthConsumerKey("DzCD85X4wVcyL6AwPtFRwDxSZ")
                .setOAuthConsumerSecret("R8oyIchRZ3Y6wpEHQRTAZ14oV3AEZIRa8eLFnMaVo543QXsFkP")
                .setOAuthAccessToken("2680166335-Qz7Amfi0bc3gYaqnFxHmmH8BJVc10vN9mAOFuLB")
                .setOAuthAccessTokenSecret("DwgayvsCmqLaEkugcsssYjf8Zqw12tfgWldq6B2FS7DkV");
        


        this.twitterStream = new TwitterStreamFactory(cb.build()).getInstance();
        this.queue = new LinkedBlockingQueue<Status>();

        final StatusListener listener = new StatusListener() {
            
			public void onStatus(Status status) {
				 //System.out.println(status.getText());
                 queue.offer(status);
            }
			
            public void onDeletionNotice(StatusDeletionNotice sdn) {
            }


            public void onTrackLimitationNotice(int i) {
            }


            public void onScrubGeo(long l, long l1) {
            }

            public void onException(Exception e) {
            }

            public void onStallWarning(StallWarning warning) {
            }

			

        };
        double[][] boundingBox= {
        		  {
        		    -180, -90
        		  }
        		  , {
        		    180, 90
        		  }
        }; /// whole world;
        double[] GEOBOX_WORLD= {-180,-90,180,90};
        double[][] GEOBOX_MELBOURNE= {{140.3482679711,-39.7125077525},{149.6598486874,-32.0871185235}};
        twitterStream.addListener(listener);
        final FilterQuery query = new FilterQuery();
        //query.track(new String[]{"a","australia","melbourne"});
        query.locations(GEOBOX_MELBOURNE);
        query.language(new String[]{"en"});
        twitterStream.filter(query);
    }

    public void nextTuple() {

        final Status status = queue.poll();
        if (status == null) {
            Utils.sleep(50);
        } else {
        	//writer.println(DataObjectFactory.getRawJSON(status));
            collector.emit(new Values(status));
        }
    }

    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("tweet"));
    }

    public void close() {
        twitterStream.shutdown();
    }

}
