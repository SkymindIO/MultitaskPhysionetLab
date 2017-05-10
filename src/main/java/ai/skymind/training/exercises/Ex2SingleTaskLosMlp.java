package ai.skymind.training.exercises;

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
 * EXERCISE 1: train an MLP to predict length of stay (LOS) using the Physionet
 * Challenge 2012 data publicly available at https://physionet.org/challenge/2012/
 *
 * The goal here is to show you how to modify the previous architecture,
 * which was designed to perform binary classification, to perform
 * multiclass classification. Our final multitask model will combine both
 * types of tasks, so this will give you insights into how to build that
 * model.
 *
 * STEPS:
 * 1) Change task specific configurations (data location, number of targets)
 * 2) Modify ETL/vectorization to treat labels as one-hot multiclass
 * 3) Modify prediction layer in model to be softmax (multiclass)
 * 4) Modify loss layer in model to be multiclass cross entropy
 * 5) Modify model evaluation to use multiclass ROC
 * 6) [optional] modify model architecture
 * 7) [optional] tune training hyperparameters
 *
 * In this example, we use hand-engineered features rather than the raw
 * time series data. Further, we use only Set A of the PC2012 data since
 * that is the only dataset for which labels are publicly available.
 *
 * Rather than use raw LOS, we convert the problem into a multiclass
 * classification task by grouping LOS into ten buckets: <2 days,
 * 2-3, 3-4, 4-5, 5-6, 6-7, 7-30, >30. These are inspired by
 * LOS buckets commonly used in LOS research.
 *
 * Author: Dave Kale (dave@skymind.io)
 */
public class Ex2SingleTaskLosMlp {
    private static File baseDir = new File("src/main/resources/physionet2012");
    private static File featuresDir = new File(baseDir, "features");

    /* STEP 1: change task-specific settings, such as location of data
     * number of targets. For LOS, they should be "los_bucket" and "8"
     * respectively.
     */
    private static File labelsDir = new File(baseDir, "mortality"); // change to los_bucket
    public static final int NB_TARGETS = 1; // change to # LOS buckets

    // For logging with SL4J
    private static final Logger log = LoggerFactory.getLogger(Ex2SingleTaskLosMlp.class);

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

    /* STEP 7 (optional): if you have time, trying changing these to
     * see how they impact training behavior and performance. Learning
     * rate is an especially critical hyperparameter.
     */
    public static final int NB_EPOCHS = 20;
    public static final int RANDOM_SEED = 1234;
    public static final double LEARNING_RATE = 0.005;
    public static final int BATCH_SIZE = 32;

    public static void main(String[] args) throws IOException, InterruptedException {

        /* STEP 2: ETL/vectorization
         *
         * Because this is a multiclass (vs. binary) classification problem, we need to
         * make some changes from the previous model:
         *
         * - change output to one-hot representation be sure
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

        // Validation (tuning) split, overs file numbers 3200-3599.
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

        // Test (held out) split, which covers file numbers 3600-3999.
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

        // Model configuration and initialization
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

                /* STEP 6 (optional): try modifying the architecture and then rerunning training
                 * to see how training behavior and performance change.
                 */
                .addLayer("dense1", new DenseLayer.Builder() // hidden layer
                                .weightInit(WeightInit.RELU)
                                .activation(Activation.RELU)
                                .nIn(NB_INPUTS)
                                .nOut(750)
                                .build(),
                        "features")

                /* STEP 3: change to the prediction layer to reflect that this is a
                 * multiclass classification problem. NB_TARGETS will change the
                 * number of outputs automatically, but we need to change the
                 * nonlinearity from SIGMOID to SOFTMAX because the classes are
                 * mutually exclusive.
                 */
                .addLayer("predictLos", new DenseLayer.Builder() // prediction layer
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SIGMOID) // change to SIGMOID
                                .nOut(NB_TARGETS)
                                .build(),
                        "dense1")

                /* STEP 4: change to the loss to reflect that this is a
                 * multiclass classification problem. We need to use multiclass
                 * cross-entropy (LossMCXENT) instead of binary (BinaryXENT).
                 */
                .addLayer("lossLos", new LossLayer.Builder(new LossBinaryXENT()) // change to LossMCXENT
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

        // Without a GUI
//        model.setListeners(new ScoreIterationListener(10));

        // Model training
        for (int epoch = 0; epoch < NB_EPOCHS; epoch++) { // outer loop over epochs
            model.fit(trainData); // implicit inner loop over minibatches


            /* STEP 5: since this is a multiclass problem, we need to compute
             * AUC for each class and average over them. Thus, we need to
             * change from basic ROC to ROCMultiClass and change the call
             * to compute AUC from calculateAUC to calculateAverageAUC.
             *
             * Don't forget to do this for all splits, including the test
             * split below!
             */
            ROC roc = new ROC(100); // change to ROCMultiClass
            trainData.reset();
            while (trainData.hasNext()) {
                MultiDataSet batch = trainData.next();
                INDArray[] output = model.output(batch.getFeatures());
                roc.eval(batch.getLabels(0), output[0]);
            }
            log.info("EPOCH " + epoch + " TRAIN AUC: " + roc.calculateAUC()); // change to calculateAverageAUC

            // loop over batches in validation data to compute validation AUC
            roc = new ROC(100); // change to ROCMultiClass
            while (validData.hasNext()) {
                MultiDataSet batch = validData.next();
                INDArray[] output = model.output(batch.getFeatures());
                roc.eval(batch.getLabels(0), output[0]);
            }
            log.info("EPOCH " + epoch + " VALID AUC: " + roc.calculateAUC()); // change to calculateAverageAUC
            trainData.reset();
            validData.reset();
        }

        // finally, loop over batches in test data to compute test AUC
        ROC roc = new ROC(100); // change to ROCMultiClass
        while (testData.hasNext()) {
            MultiDataSet batch = testData.next();
            INDArray[] output = model.output(batch.getFeatures());
            roc.eval(batch.getLabels(0), output[0]);
        }
        testData.reset();
        log.info("TEST AUC: " + roc.calculateAUC()); // change to calculateAverageAUC
    }
}
