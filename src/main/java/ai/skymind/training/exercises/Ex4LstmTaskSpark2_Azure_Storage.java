package ai.skymind.training.exercises;

import org.deeplearning4j.eval.ROC;
import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.LearningRatePolicy;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.writable.Writable;
import org.datavec.spark.functions.pairdata.BytesPairWritable;
import org.datavec.spark.functions.pairdata.PairSequenceRecordReaderBytesFunction;
import org.datavec.spark.functions.pairdata.PathToKeyConverter;
import org.datavec.spark.functions.pairdata.PathToKeyConverterFilename;
import org.datavec.spark.util.DataVecSparkUtil;
import org.deeplearning4j.spark.datavec.DataVecSequencePairDataSetFunction;
import org.nd4j.linalg.dataset.DataSet;
import scala.Tuple2;

import java.util.List;
import org.apache.hadoop.conf.Configuration;

/**
 * EXERCISE 4: train a LSTM using spark to predict mortality using the Physionet
 * Challenge 2012 data publicly available at https://physionet.org/challenge/2012/
 *
 *
 */

public class Ex4LstmTaskSpark2_Azure_Storage {

    // For logging with SL4J
    private static final Logger log = LoggerFactory.getLogger(Ex4LstmTaskSpark2_Azure_Storage.class);

    // Number of training, validation, test examples
    public static final int NB_TRAIN_EXAMPLES = 3200;
    public static final int NB_VALID_EXAMPLES = 400;
    public static final int NB_TEST_EXAMPLES = 4000 - NB_TRAIN_EXAMPLES - NB_VALID_EXAMPLES;

    // Number of features (inputs), heart rate, blood pressure
    public static final int NB_INPUTS = 86;

    public static final int NB_EPOCHS = 25;
    public static final int RANDOM_SEED = 1234;
    public static final double LEARNING_RATE = 0.032;
    public static final int BATCH_SIZE = 40;
    public static final int lstmLayerSize = 200;    //Number of units in each GravesLSTM layer


    public static void main(String[] args) throws IOException, InterruptedException {

        // Set up Spark Context

        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("Spark Lstm example");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // Set Configuration (not sure if needed)

        Configuration config = sc.hadoopConfiguration();
        config.set("fs.azure", "org.apache.hadoop.fs.azure.NativeAzureFileSystem");
       // config.set("fs.azure.account.key.bparksa.blob.core.windows.net", "NNR7VpHNDKWXzXFyMhclsx2foxyyCXnrHE6UXJiom6ARcKi2s90GbwMj69E0inC/IWv5QzD8Ovrrv3VY8AK/cA==");

        // Path to sequence and mortality directories via Azure blob container
        String featuresPath = "wasb:///resourcesP/physionet2012/sequence";
        String labelsPath = "wasb:///resourcesP/physionet2012/mortality";

        // Convert data to JavaRDD<DataSet>

        JavaPairRDD<String, PortableDataStream> featureFiles = sc.binaryFiles(featuresPath);
        JavaPairRDD<String, PortableDataStream> labelFiles = sc.binaryFiles(labelsPath);

        PathToKeyConverter pathToKeyConverter = new PathToKeyConverterFilename();   //new PathToKeyConverterNumber();
        JavaPairRDD<Text, BytesPairWritable> rdd = DataVecSparkUtil.combineFilesForSequenceFile(sc, featuresPath, labelsPath, pathToKeyConverter);

        SequenceRecordReader srr1 = new CSVSequenceRecordReader();
        SequenceRecordReader srr2 = new CSVSequenceRecordReader();
        PairSequenceRecordReaderBytesFunction fn = new PairSequenceRecordReaderBytesFunction(srr1, srr2);
        JavaRDD<Tuple2<List<List<Writable>>, List<List<Writable>>>> writables = rdd.map(fn);

        int nClasses = 2;
        boolean regression = false;
        JavaRDD<DataSet> dataSets = writables.map(new DataVecSequencePairDataSetFunction(nClasses, regression, DataVecSequencePairDataSetFunction.AlignmentMode.ALIGN_END));

        // For Debugging
        System.out.println("Breakpoint 1");

        // Build Computation Graph

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(RANDOM_SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(LEARNING_RATE)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.ADAM)
                .dropOut(0.25)
                .graphBuilder()
                .addInputs("trainFeatures")
                .setOutputs("predictMortality")
                .addLayer("L1", new GravesLSTM.Builder()
                                .nIn(NB_INPUTS)
                                .nOut(lstmLayerSize)
                                .forgetGateBiasInit(1)
                                .activation(Activation.TANH)
                                .build(),
                        "trainFeatures")
                .addLayer("predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(lstmLayerSize).nOut(nClasses).build(),"L1")
                .pretrain(false).backprop(true)
                .build();

        // Set Training Master

        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1)
                .averagingFrequency(5) //This controls how frequently the parameters are averaged and redistributed, in terms of number of minibatches of size batchSizePerWorker
                .workerPrefetchNumBatches(2) // avoid data loading time
                .batchSizePerWorker(BATCH_SIZE) //it is the number of examples used for each parameter update in each worker.
                .build();

        // Initialize SparkComputationGraph

        SparkComputationGraph sparkNet = new SparkComputationGraph(sc, conf, tm);

        // For Debugging

        System.out.println("Breakpoint 2");

        // ERROR
        sparkNet.fit(dataSets);

        // For Debugging

        System.out.println("Breakpoint 3");
    }
}