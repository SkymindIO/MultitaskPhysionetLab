package ai.skymind.training.exercises;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.eval.ROC;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;


/**
 * EXERCISE 1: train an MLP to predict mortality using the Physionet
 * Challenge 2012 data publicly available at https://physionet.org/challenge/2012/
 *
 * The goal here is to walk through some of the common DL4J patterns (ETL,
 * vectorization, training, evaluation) and get training a model as quickly
 * as possible. As such, you'll do very little coding, except if you want
 * to try to experiment with the model architecture.
 *
 * STEPS:
 * 1) Learn ETL/vectorization
 * 2) Learn model configuration
 * 3) Learn performance monitoring and visualization
 * 4) Learn model training
 * 5) Learn model evaluation
 * 6) [optional] modify model architecture
 * 7) [optional] tune training hyperparameters
 *
 * In this example, we use hand-engineered features rather than the raw
 * time series data. Further, we use only Set A of the PC2012 data since
 * that is the only dataset for which labels are publicly available.
 *
 * We use the in-hospital mortality targets as is.
 *
 * Author: Dave Kale (dave@skymind.io)
 */
public class Ex1SingleTaskMortalityMlp {
    private static File baseDir = new File("src/main/resources/physionet2012");
    private static File featuresDir = new File(baseDir, "features");

    /* Task-specific configuration */
    private static File labelsDir = new File(baseDir, "mortality");
    public static final int NB_TARGETS = 1; // mortality, which is binary

    // For logging with SL4J
    private static final Logger log = LoggerFactory.getLogger(Ex1SingleTaskMortalityMlp.class);

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

        /* STEP 1: ETL/vectorization
         *
         * Look close closely at how we handle data and perform ETL and vectorization.
         * You'll be asked to extend this to handle multiple label sets (for multitask
         * learning) in Exercise 3.
         *
         * A common DL4J/DataVec ETL pattern: each record is stored in CSV format in
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
                .addOutput("trainLabels")
                .build();

        /* IMPORTANT!
         *
         * It's critical to monitor model performance on non-training data (data
         * not used for fitting weights) to identify and combat overfitting. The
         * validation (tuning) set can also be used for choosing hyperparameters
         * (e.g., learning rate), for early stopping, etc.
         *
         * Validation (tuning) split, which covers file numbers 3200-3599.
         */
        CSVRecordReader validFeatures = new CSVRecordReader();
        validFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES - 1));
        CSVRecordReader validLabels = new CSVRecordReader();
        validLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES - 1));
        MultiDataSetIterator validData = new RecordReaderMultiDataSetIterator.Builder(BATCH_SIZE)
                .addReader("validFeatures", validFeatures)
                .addInput("validFeatures")
                .addReader("validLabels", validLabels)
                .addOutput("validLabels")
                .build();

        /* IMPORTANT!
         *
         * What we ultimate care about is our model's ability to generalize about
         * future new examples (patients). To assess this ahead of time, we should
         * evaluate its performance on a held out (test) set that the model NEVER
         * sees during training. You will notice that model performance on the test
         * set is frequently worse than on training or validation data.
         *
         * In real world settings, it is critical that you use the test as little
         * as possible and that you do not make modeling decisions based on test
         * set performance. Otherwise, it becomes a de facto second validation set.
         * In practice with toy problems, you will inevitably touch the test set
         * so at a minimum, be aware of the pitfalls and be sober about your results.
         *
         * Test (held out evaluation) split, which covers file numbers 3600-3999.
         */
        CSVRecordReader testFeatures = new CSVRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
        CSVRecordReader testLabels = new CSVRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
        MultiDataSetIterator testData = new RecordReaderMultiDataSetIterator.Builder(BATCH_SIZE)
                .addReader("testFeatures", testFeatures)
                .addInput("testFeatures")
                .addReader("testLabels", testLabels)
                .addOutput("testLabels")
                .build();

        /* STEP 2: Model configuration and initialization
         *
         * In DL4J, we first configure the model, then we construct and
         * initialize the actual model (including allocation of weights, etc.).
         * This pattern is similar to other frameworks with a two-step "Configure,
         * Compile" process.
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
                .addInputs("trainFeatures") // names of input layers
                .setInputTypes(InputType.feedForward(NB_INPUTS)) // type of input

                /* NOTE: we set output names you'll need to change this in future examples. */

                .setOutputs("lossMortality") // names of outputs (i.e., where loss is compute)

                /* STEP 6 (optional): try modifying the architecture and then rerunning training
                 * to see how training behavior and performance change.
                 */
                .addLayer("dense1", new DenseLayer.Builder() // hidden layer
                                .weightInit(WeightInit.RELU)
                                .activation(Activation.RELU)
                                .nIn(NB_INPUTS)
                                .nOut(500)
                                .build(),
                        "trainFeatures")

                /* NOTE: here are the output (prediction and loss) layers. These will need
                 * to be changed in other exercises.
                 */
                .addLayer("predictMortality", new DenseLayer.Builder() // prediction layer
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SIGMOID)
                                .nOut(NB_TARGETS)
                                .build(),
                        "dense1")
                .addLayer("lossMortality", new LossLayer.Builder(new LossBinaryXENT()) // loss
                                .build(),
                        "predictMortality")
                .build();

        // Construct and initialize model
        ComputationGraph model = new ComputationGraph(config);
        model.init();

        /* STEP 3: Performance monitoring with a GUI
         *
         * This code adds some listeners for model performance and optionally
         * deploys a Play-based GUI for monitoring training. This is very useful
         * for tracking progress, debugging, etc.
         *
         * If you don't want to or can't deploy a GUI, comment this out
         * and use the simpler set of listeners below.
         */
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new ScoreIterationListener(10),
                           new StatsListener(statsStorage));

        // STEP 3 without a GUI
//        model.setListeners(new ScoreIterationListener(10));

        /* STEP 4: Model training
         *
         * Model training involves two loops, an outer loop over epochs
         * and an inner loop over minibatches. In DL4J, the inner loop
         * can be implicit (handled inside the call to fit, as shown
         * below) or explicit, as is demonstrated in the validation
         * loop below.
         */
        for (int epoch = 0; epoch < NB_EPOCHS; epoch++) { // outer loop over epochs
            model.fit(trainData); // implicit inner loop over minibatches

            // loop over batches in training data to compute training AUC
            ROC roc = new ROC(100);
            trainData.reset();
            while (trainData.hasNext()) {
                MultiDataSet batch = trainData.next();
                INDArray[] output = model.output(batch.getFeatures());
                roc.eval(batch.getLabels(0), output[0]);
            }
            log.info("EPOCH " + epoch + " TRAIN AUC: " + roc.calculateAUC());

            // loop over batches in validation data to compute validation AUC
            roc = new ROC(100);
            while (validData.hasNext()) {
                MultiDataSet batch = validData.next();
                INDArray[] output = model.output(batch.getFeatures());
                roc.eval(batch.getLabels(0), output[0]);
            }
            log.info("EPOCH " + epoch + " VALID AUC: " + roc.calculateAUC());
            trainData.reset();
            validData.reset();
        }

        /* STEP 5: Model evaluation on held out (test) data */
        // finally, loop over batches in test data to compute test AUC
        ROC roc = new ROC(100);
        while (testData.hasNext()) {
            MultiDataSet batch = testData.next();
            INDArray[] output = model.output(batch.getFeatures());
            roc.eval(batch.getLabels(0), output[0]);
        }
        testData.reset();
        log.info("FINAL TEST AUC: " + roc.calculateAUC());
    }
}
