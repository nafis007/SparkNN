package com.mycompany.singlespark;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import backtype.storm.spout.SpoutOutputCollector;
import backtype.storm.task.TopologyContext;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.base.BaseRichSpout;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Values;

public class fileReaderSpout extends BaseRichSpout {

    private SpoutOutputCollector collector;
    private boolean completed = false;
    private FileReader fileReader;
    private String str;
    private BufferedReader reader ;
    private int count;



    public void open(Map conf, TopologyContext context, SpoutOutputCollector collector) {
        try {
            this.fileReader = new FileReader(conf.get("initialData").toString());
        } catch (FileNotFoundException e) {
            throw new RuntimeException("Error reading file ["+conf.get("wordFile")+"]");
        }

        this.collector = collector;
        this.reader =  new BufferedReader(fileReader);
        this.count = 1;
    }


    public void nextTuple() {

        if (!completed) {
          try {

                this.str = reader.readLine();
                if (this.str != null) {
                	String[] row = str.split(",");
                    this.collector.emit(new Values(count,row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],
                    						row[11],row[12],row[13],row[14],row[15],row[16],row[17],row[18],row[19],row[20],
                    						row[21],row[22],row[23],row[24],row[25],row[26],row[27],row[28],row[29],row[30],
                    						row[31],row[32],row[33],row[34],row[35],row[36],row[37],row[38],row[39],row[40],
                    						row[41],row[42],row[43],row[44],row[45],row[46],row[47],row[48],row[49],row[50],
                    						row[51],row[52],row[53],row[54],row[55],row[56],row[57],row[58],row[59],row[60],
                    						row[61],row[62],row[63],row[64],row[65],row[66],row[67],row[68],row[69],row[70],
                    						row[71],row[72],row[73],row[74],row[75],row[76],row[77],row[78],row[79],row[80],
                    						row[81],row[82],row[83],row[84],row[85],row[86],row[87],row[88],row[89],row[90],
                    						row[91],row[92],row[93],row[94],row[95],row[96],row[97],row[98],row[99],row[100]));
                    count++;
                } else {
                    completed = true;
                    fileReader.close();;
                }

            } catch (Exception e) {
                throw new RuntimeException("Error reading tuple", e);
            }
        }
    }

    public void declareOutputFields(OutputFieldsDeclarer declarer) {
    	//For Most 100
    	//declarer.declare(new Fields("count","auspol","follow","ssm","de","australia","equal","vote","die","survey","ich","thisisaustralia","lnp","springst","countri","marriag","votey","und","pleas","der","love","futur","time","peopl","abbott","right","en","la","win","past","das","ensur","day","je","que","yes","ist","free","look","artist","am","ein","thank","nicht","auf","play","tweet","es","le","support","job","un","mit","re","join","marriageequ","du","im","comment","via","govern","australian","call","melbourn","toni","den","von","et","live","turnbul","pas","na","govt","mal","polit","chang","al","des","hope","insid","labor","auch","do","les","een","ik","se","oh","zu","berlin","countrymus","aus","da","postal","thx","fuck","believ","parti","start","plebiscit","tri","location"));
    	//For Top 50
        declarer.declare(new Fields("count","auspol","follow","ssm","australia","equal","vote","survey","thisisaustralia","springst","lnp","votey","marriag","countri","pleas","futur","right","abbott","past","ensur","win","peopl","time","artist","yes","free","day","play","love","look","marriageequ","support","join","tweet","govern","australian","thank","comment","melbourn","toni","turnbul","govt","labor","countrymus","job","polit","insid","postal","call","thx","jeffcrew","de","die","ich","und","der","en","la","das","je","que","ist","ein","nicht","auf","es","le","mit","un","loves","du","am","im","von","den","et","pas","na","des","auch","ik","een","les","berlin","se","zu","al","mal","da","te","noch","het","van","ja","bei","times","aber","il","ne","wie","days","location"));
    }

    public void ack(Object msgId) {}


    public void fail(Object msgId) {}
}
