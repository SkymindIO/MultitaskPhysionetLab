package ai.skymind.training.solutions;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.eval.ROC;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;


/**
 * Trains a MLP to predict length of stay (LOS) using the Physionet Challenge
 * 2012 data publicly available at https://physionet.org/challenge/2012/
 *
 * In this example, we use hand-engineered features rather than the raw
 * time series data. Further, we use only Set A of the PC2012 data since
 * that is the only dataset for which labels are publicly available.
 *
 * Rather than use raw LOS, we convert the problem into a multiclass
 * classification task by grouping LOS into ten buckets: <1 day, 1-2 days,
 * 2-3, 3-4, 4-5, 5-6, 6-7, 7-14, 14-30, >30. These are inspired by
 * LOS buckets commonly used in LOS research.
 *
 * Author: Dave Kale (dave@skymind.io)
 */
public class SingleTaskLosModel {
    // For logging with SL4J
    private static final Logger log = LoggerFactory.getLogger(SingleTaskLosModel.class);

    // Number of training, validation, test examples
    public static final int NB_TRAIN_EXAMPLES = 3200;
    public static final int NB_VALID_EXAMPLES = 400;
    public static final int NB_TEST_EXAMPLES = 4000 - NB_TRAIN_EXAMPLES - NB_VALID_EXAMPLES;

    /* Number of features (inputs):
     * - 36 variables x 9 statistics
     * - ICU Type is categorical with 4 values
     * - Mortality is categorical with 2 values (but both can be turned off)
     * - age, height, weight
     */
    public static final int NB_INPUTS = 36 * 9 + 4 + 2 + 3 * 1;
    public static final int NB_TARGETS = 8; // LOS bucket

    // These control learning
    public static final int NB_EPOCHS = 20;
    public static final int RANDOM_SEED = 1234;
    public static final double LEARNING_RATE = 0.005;
    public static final int BATCH_SIZE = 32;

    private static File baseDir = new File("src/main/resources/physionet2012");
    private static File featuresDir = new File(baseDir, "features");
    private static File labelsDir = new File(baseDir, "los_bucket");

    public static void main(String[] args) throws IOException, InterruptedException {

        /* A common DL4J/DataVec ETL pattern: each record is stored in CSV format in
         * a distinct numbered file, and features and labels are stored separately.
         *
         * We first create NumberedFileSplit and pass in the range of file numbers
         * that should be processed. We create a CSVRecordReader (these data are stored
         * in CSV format) and initialize it using the split. Finally, we pass the record
         * readers to a data set iterator. In this case, since features and labels are
         * stored separately, we use a multi-dataset iterator.
         *
         * This is for the training split, which covers file numbers 0-3199.
         */
        NumberedFileInputSplit trainFeaturesSplit = new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1);
        CSVRecordReader trainFeatures = new CSVRecordReader();
        trainFeatures.initialize(trainFeaturesSplit);
        NumberedFileInputSplit trainLabelsSplit = new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1);
        CSVRecordReader trainLabels = new CSVRecordReader();
        trainLabels.initialize(trainLabelsSplit);
        MultiDataSetIterator trainData = new RecordReaderMultiDataSetIterator.Builder(BATCH_SIZE)
                .addReader("trainFeatures", trainFeatures)
                .addInput("trainFeatures")
                .addReader("trainLabels", trainLabels)
                .addOutputOneHot("trainLabels", 0, NB_TARGETS)
                .build();

        // Validation (tuning) split, which covers file numbers 3200-3599.
        CSVRecordReader validFeatures = new CSVRecordReader();
        validFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES - 1));
        CSVRecordReader validLabels = new CSVRecordReader();
        validLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES - 1));
        MultiDataSetIterator validData = new RecordReaderMultiDataSetIterator.Builder(BATCH_SIZE)
                .addReader("validFeatures", validFeatures)
                .addInput("validFeatures")
                .addReader("validLabels", validLabels)
                .addOutputOneHot("validLabels", 0, NB_TARGETS)
                .build();

        // Validation (tuning) split, which covers file numbers 3200-3599.
        CSVRecordReader testFeatures = new CSVRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
        CSVRecordReader testLabels = new CSVRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
        MultiDataSetIterator testData = new RecordReaderMultiDataSetIterator.Builder(BATCH_SIZE)
                .addReader("testFeatures", testFeatures)
                .addInput("testFeatures")
                .addReader("testLabels", testLabels)
                .addOutputOneHot("testLabels", 0, NB_TARGETS)
                .build();

        /* In DL4J, we first configure the model, then we construct and
         * initialize the actual model (including allocation of weights, etc.)
         *
         * We are using the ComputationGraph (CG) architecture, which is more
         * flexible than the MultiLayerNetwork (MLN). The CG allows multiple
         * inputs, multiple outputs (for, e.g., multitask learning), and multiple
         * paths. However, it is a little more complex to understand and configure.
         *
         * In DL4J models, we typically set global (across all layers) configurations
         * first, including things like random seed and optimization, then we add
         * layers one at a time.
         */
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(RANDOM_SEED) // fixing the random seed ensures reproducibility
                .iterations(1) // ignored for SGD
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // whether to use SGD, LBFGS, etc.
                .learningRate(LEARNING_RATE) // basic learning rate
                .updater(Updater.ADAM) // modified (SGD) update rule, e.g., Nesterov, ADAM, ADADELTA, etc.
                .graphBuilder() // constructs a ComputationGraph
                .addInputs("features") // names of input layers
                .setInputTypes(InputType.feedForward(NB_INPUTS)) // type of input
                .setOutputs("lossLos") // names of outputs (i.e., where loss is compute)
                .addLayer("dense1", new DenseLayer.Builder() // hidden layer
                                .weightInit(WeightInit.RELU)
                                .activation(Activation.RELU)
                                .nIn(NB_INPUTS)
                                .nOut(750)
                                .build(),
                        "features")
                .addLayer("predictLos", new DenseLayer.Builder() // prediction layer
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX)
                                .nOut(NB_TARGETS)
                                .build(),
                        "dense1")
                .addLayer("lossLos", new LossLayer.Builder(new LossMCXENT()) // loss
                                .build(),
                        "predictLos")
                .build();

        // Construct and initialize model
        ComputationGraph model = new ComputationGraph(config);
        model.init();

        /* This code sets up a Play-based GUI for monitoring training.
         * This is very useful for tracking progress, debugging, etc.
         */
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new ScoreIterationListener(10),
                new StatsListener(statsStorage),
                new PerformanceListener(10));

        /* Model training involves two loops, an outer loop over epochs
         * and an inner loop over minibatches. In DL4J, the inner loop
         * can be implicit (handled inside the call to fit, as shown
         * below) or explicit, as is demonstrated in the validation
         * loop below.
         */
        for (int epoch = 0; epoch < NB_EPOCHS; epoch++) { // outer loop over epochs
            model.fit(trainData); // implicit inner loop over minibatches

            // loop over batches in training data to compute training AUC
            ROCMultiClass roc = new ROCMultiClass(100);
            trainData.reset();
            while (trainData.hasNext()) {
                MultiDataSet batch = trainData.next();
                INDArray[] output = model.output(batch.getFeatures());
                roc.eval(batch.getLabels(0), output[0]);
            }
            log.info("EPOCH " + epoch + " TRAIN AVG AUC: " + roc.calculateAverageAUC());

            // loop over batches in validation data to compute validation AUC
            roc = new ROCMultiClass(100);
            while (validData.hasNext()) {
                MultiDataSet batch = validData.next();
                INDArray[] output = model.output(batch.getFeatures());
                roc.eval(batch.getLabels(0), output[0]);
            }
            log.info("EPOCH " + epoch + " VALID AVG AUC: " + roc.calculateAverageAUC());
            trainData.reset();
            validData.reset();
        }

        // finally, loop over batches in test data to compute test AUC
        ROCMultiClass roc = new ROCMultiClass(100);
        while (testData.hasNext()) {
            MultiDataSet batch = testData.next();
            INDArray[] output = model.output(batch.getFeatures());
            roc.eval(batch.getLabels(0), output[0]);
        }
        testData.reset();
        log.info("TEST AVG AUC: " + roc.calculateAverageAUC());
    }
}
