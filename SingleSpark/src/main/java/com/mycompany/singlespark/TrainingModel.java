package com.mycompany.singlespark;

import org.datavec.api.records.metadata.RecordMetaData;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.nd4j.linalg.dataset.DataSet;

import java.util.List;

public class TrainingModel {
    SparkDl4jMultiLayer sparkNetwork;
    DataSet testDataSet;
    List<RecordMetaData> testMetaData;

    public TrainingModel(SparkDl4jMultiLayer sparkNetwork, DataSet testDataSet, List<RecordMetaData> testMetaData) {
        this.sparkNetwork = sparkNetwork;
        this.testDataSet = testDataSet;
        this.testMetaData = testMetaData;
    }

}
