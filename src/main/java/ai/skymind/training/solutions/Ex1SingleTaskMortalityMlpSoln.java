package ai.skymind.training.solutions;

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
 * Trains a MLP to predict mortality using the Physionet Challenge 2012
 * data publicly available at https://physionet.org/challenge/2012/
 *
 * In this example, we use hand-engineered features rather than the raw
 * time series data. Further, we use only Set A of the PC2012 data since
 * that is the only dataset for which labels are publicly available.
 *
 * We use the in-hospital mortality targets as is.
 *
 * Author: Dave Kale (dave@skymind.io)
 */
public class Ex1SingleTaskMortalityMlpSoln {
    // For logging with SL4J
    private static final Logger log = LoggerFactory.getLogger(Ex1SingleTaskMortalityMlpSoln.class);

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
    public static final int NB_TARGETS = 1; // mortality, which is binary

    // These control learning
    public static final int NB_EPOCHS = 20;
    public static final int RANDOM_SEED = 1234;
    public static final double LEARNING_RATE = 0.005;
    public static final int BATCH_SIZE = 32;

    private static File baseDir = new File("src/main/resources/physionet2012");
    private static File featuresDir = new File(baseDir, "features");
    private static File labelsDir = new File(baseDir, "mortality");

    public static void main(String[] args) throws IOException, InterruptedException {

        // Training split, which covers file numbers 0-3199.
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

        // Validation (tuning) split, which covers file numbers 3200-3599.
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

        // Test (held out) split, which covers file numbers 3600-3999.
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

        // Model configuration and initialization
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(RANDOM_SEED) // fixing the random seed ensures reproducibility
                .iterations(1) // ignored for SGD
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // whether to use SGD, LBFGS, etc.
                .learningRate(LEARNING_RATE) // basic learning rate
                .updater(Updater.ADAM) // modified (SGD) update rule, e.g., Nesterov, ADAM, ADADELTA, etc.
                .graphBuilder() // constructs a ComputationGraph
                .addInputs("trainFeatures") // names of input layers
                .setInputTypes(InputType.feedForward(NB_INPUTS)) // type of input
                .setOutputs("lossMortality") // names of outputs (i.e., where loss is compute)
                .addLayer("dense1", new DenseLayer.Builder() // hidden layer
                                .weightInit(WeightInit.RELU)
                                .activation(Activation.RELU)
                                .nIn(NB_INPUTS)
                                .nOut(500)
                                .build(),
                        "trainFeatures")
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

        /* This code sets up a Play-based GUI for monitoring training.
         * This is very useful for tracking progress, debugging, etc.
         */
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new ScoreIterationListener(10),
                new StatsListener(statsStorage),
                new PerformanceListener(10));

        // No GUI
//        model.setListeners(new ScoreIterationListener(10));

        // Model training
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

        // Model evaluation: loop over batches in test data to compute test AUC
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
