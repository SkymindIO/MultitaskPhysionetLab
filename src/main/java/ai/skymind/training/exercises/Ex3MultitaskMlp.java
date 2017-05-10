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
 * Trains a MLP to predict mortality, length of stay (LOS), and survival
 * using the Physionet Challenge 2012 data publicly available at https://physionet.org/challenge/2012/
 *
 * STEPS:
 * 1) Add task-specific configurations (data location, number of targets)
 * 2) Modify ETL/vectorization to handle multiple task labels
 * 3) Add output layer names for additional tasks to model
 * 4) Add prediction and loss layers for additional tasks to model
 * 5) Add ROC evaluations for each task
 * 6) [optional] modify model architecture
 * 7) [optional] tune training hyperparameters
 *
 * This is a multitask MLP with three task-specific outputs.
 *
 * Author: Dave Kale (dave@skymind.io)
 */
public class Ex3MultitaskMlp {
    private static File baseDir = new File("src/main/resources/physionet2012");
    private static File featuresDir = new File(baseDir, "features");

    /* STEP 1: Multitask configuration. We start with mortality but add
     * LOS (8 targets, los_bucket) and optionally survival (5 targets,
     * survival_bucket).
     */
    private static File mortalityDir = new File(baseDir, "mortality");
    public static final int NB_MORTALITY_TARGETS = 1; // mortality
//    private static File losDir = ... // should be los_bucket
//    public static final int NB_LOS_TARGETS = ... // should be 8
//    private static File survivalDir = ... // should be survival_bucket
//    public static final int NB_SURVIVAL_TARGETS = ... // should be 5

    // For logging with SL4J
    private static final Logger log = LoggerFactory.getLogger(Ex3MultitaskMlp.class);

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

        /* STEP 2: add the additional task label data sets to the iterator.
         * You will need to configure a separate record reader for each set
         * of labels, add each reader to the multidataset iterator independently,
         * and configure the outputs appropriately. Note that LOS and survival
         * are multiclass so they should be one-hot. ;)
         */
        CSVRecordReader trainFeatures = new CSVRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));
        CSVRecordReader trainMortalityLabels = new CSVRecordReader();
        trainMortalityLabels.initialize(new NumberedFileInputSplit(mortalityDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));
//        CSVRecordReader trainLosLabels = new CSVRecordReader();
//        trainLosLabels.initialize(...);
//        CSVRecordReader trainSurvivalLabels = new CSVRecordReader();
//        trainSurvivalLabels.initialize(...);
        MultiDataSetIterator trainData = new RecordReaderMultiDataSetIterator.Builder(BATCH_SIZE)
                .addReader("trainFeatures", trainFeatures)
                .addInput("trainFeatures")
                .addReader("trainMortalityLabels", trainMortalityLabels)
                .addOutput("trainMortalityLabels")
                /* IMPORTANT! Add readers and outputs for other tasks! */
                .build();

        // Validation (tuning) split, which covers file numbers 3200-3599.
        CSVRecordReader validFeatures = new CSVRecordReader();
        validFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES - 1));
        CSVRecordReader validMortalityLabels = new CSVRecordReader();
        validMortalityLabels.initialize(new NumberedFileInputSplit(mortalityDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES - 1));
//        CSVRecordReader validLosLabels = new CSVRecordReader();
//        validLosLabels.initialize(...);
//        CSVRecordReader validSurvivalLabels = new CSVRecordReader();
//        validSurvivalLabels.initialize(...);
        MultiDataSetIterator validData = new RecordReaderMultiDataSetIterator.Builder(BATCH_SIZE)
                .addReader("validFeatures", validFeatures)
                .addInput("validFeatures")
                .addReader("validMortalityLabels", validMortalityLabels)
                .addOutput("validMortalityLabels")
                /* IMPORTANT! Add readers and outputs for other tasks! */
                .build();

        // Validation (tuning) split, which covers file numbers 3200-3599.
        CSVRecordReader testFeatures = new CSVRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
        CSVRecordReader testMortalityLabels = new CSVRecordReader();
        testMortalityLabels.initialize(new NumberedFileInputSplit(mortalityDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
//        CSVRecordReader testLosLabels = new CSVRecordReader();
//        testLosLabels.initialize();
//        CSVRecordReader testSurvivalLabels = new CSVRecordReader();
//        testSurvivalLabels.initialize();
        MultiDataSetIterator testData = new RecordReaderMultiDataSetIterator.Builder(BATCH_SIZE)
                .addReader("testFeatures", testFeatures)
                .addInput("testFeatures")
                .addReader("testMortalityLabels", testMortalityLabels)
                .addOutput("testMortalityLabels")
                /* IMPORTANT! Add readers and outputs for other tasks! */
                .build();

        /* STEPS 3-4: Configure multitask neural net.
         * We start with the basic mortality MLP and add additional outputs for
         * LOS and (optionally) survival. There are several changes to be made
         * so look carefully:
         *
         *
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

                /* STEP 3: add additional outputs here, i.e., setOutputs("lossMortality", "lossLos", ...) */
                .setOutputs("lossMortality") // names of outputs (i.e., where loss is compute)

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
                .addLayer("predictMortality", new DenseLayer.Builder() // prediction layer
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SIGMOID)
                                .nOut(NB_MORTALITY_TARGETS)
                                .build(),
                        "dense1")
                .addLayer("lossMortality", new LossLayer.Builder(new LossBinaryXENT()) // loss
                                .build(),
                        "predictMortality")

                /* STEP 4: add additional prediction and loss layers here for LOS and
                 * optionally survival, i.e.,
                 *
                 *   .addLayer("predictLos", ..., "dense1")
                 *   .addLayer("lossLos", ..., "predictLos")
                 *
                 * Don't forget that LOS and survival are multiclass problems!
                 */
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

            /* STEP 5: add ROCMultiClass evaluations here for the additional
             * LOS and (optional) survival tasks that you've added.
             *
             * Don't forget to do this for all splits, including the test
             * split below!
             */
            ROC rocMortality = new ROC(100);
            trainData.reset();
            while (trainData.hasNext()) {
                MultiDataSet batch = trainData.next();
                INDArray[] output = model.output(batch.getFeatures());
                rocMortality.eval(batch.getLabels(0), output[0]);
            }
            log.info("EPOCH " + epoch + " TRAIN MORTALITY AUC: " + rocMortality.calculateAUC());

            // loop over batches in validation data to compute validation AUC
            rocMortality = new ROC(100);
            while (validData.hasNext()) {
                MultiDataSet batch = validData.next();
                INDArray[] output = model.output(batch.getFeatures());
                rocMortality.eval(batch.getLabels(0), output[0]);
            }
            log.info("EPOCH " + epoch + " VALID MORTALITY AUC: " + rocMortality.calculateAUC());
            trainData.reset();
            validData.reset();
        }

        // finally, loop over batches in test data to compute test AUC
        ROC rocMortality = new ROC(100);
        trainData.reset();
        while (trainData.hasNext()) {
            MultiDataSet batch = trainData.next();
            INDArray[] output = model.output(batch.getFeatures());
            rocMortality.eval(batch.getLabels(0), output[0]);
        }
        log.info("FINAL TEST MORTALITY AUC: " + rocMortality.calculateAUC());
    }
}
