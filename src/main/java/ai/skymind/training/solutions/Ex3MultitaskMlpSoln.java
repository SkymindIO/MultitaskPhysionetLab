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
 * Trains a MLP to predict mortality, length of stay (LOS), and survival
 * using the Physionet Challenge 2012 data publicly available at https://physionet.org/challenge/2012/
 *
 * This is a multitask MLP with three task-specific outputs.
 *
 * Author: Dave Kale (dave@skymind.io)
 */
public class Ex3MultitaskMlpSoln {
    // For logging with SL4J
    private static final Logger log = LoggerFactory.getLogger(Ex3MultitaskMlpSoln.class);

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
    public static final int NB_MORTALITY_TARGETS = 1; // mortality
    public static final int NB_LOS_TARGETS = 8; // LOS bucket
    public static final int NB_SURVIVAL_TARGETS = 5; // survival bucket

    // These control learning
    public static final int NB_EPOCHS = 20;
    public static final int RANDOM_SEED = 1234;
    public static final double LEARNING_RATE = 0.005;
    public static final int BATCH_SIZE = 32;

    private static File baseDir = new File("src/main/resources/physionet2012");
    private static File featuresDir = new File(baseDir, "features");
    private static File mortalityDir = new File(baseDir, "mortality");
    private static File losDir = new File(baseDir, "los_bucket");
    private static File survivalDir = new File(baseDir, "survival_bucket");

    public static void main(String[] args) throws IOException, InterruptedException {

        // Training split, which covers file numbers 0-3199.
        CSVRecordReader trainFeatures = new CSVRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));
        CSVRecordReader trainMortalityLabels = new CSVRecordReader();
        trainMortalityLabels.initialize(new NumberedFileInputSplit(mortalityDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));
        CSVRecordReader trainLosLabels = new CSVRecordReader();
        trainLosLabels.initialize(new NumberedFileInputSplit(losDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));
        CSVRecordReader trainSurvivalLabels = new CSVRecordReader();
        trainSurvivalLabels.initialize(new NumberedFileInputSplit(survivalDir.getAbsolutePath() + "/%d.csv", 0, NB_TRAIN_EXAMPLES - 1));
        MultiDataSetIterator trainData = new RecordReaderMultiDataSetIterator.Builder(BATCH_SIZE)
                .addReader("trainFeatures", trainFeatures)
                .addInput("trainFeatures")
                .addReader("trainMortalityLabels", trainMortalityLabels)
                .addOutput("trainMortalityLabels")
                .addReader("trainLosLabels", trainLosLabels)
                .addOutputOneHot("trainLosLabels", 0, NB_LOS_TARGETS)
                .addReader("trainSurvivalLabels", trainSurvivalLabels)
                .addOutputOneHot("trainSurvivalLabels", 0, NB_SURVIVAL_TARGETS)
                .build();

        // Validation (tuning) split, which covers file numbers 3200-3599.
        CSVRecordReader validFeatures = new CSVRecordReader();
        validFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES - 1));
        CSVRecordReader validMortalityLabels = new CSVRecordReader();
        validMortalityLabels.initialize(new NumberedFileInputSplit(mortalityDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES - 1));
        CSVRecordReader validLosLabels = new CSVRecordReader();
        validLosLabels.initialize(new NumberedFileInputSplit(losDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES - 1));
        CSVRecordReader validSurvivalLabels = new CSVRecordReader();
        validSurvivalLabels.initialize(new NumberedFileInputSplit(survivalDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES - 1));
        MultiDataSetIterator validData = new RecordReaderMultiDataSetIterator.Builder(BATCH_SIZE)
                .addReader("validFeatures", validFeatures)
                .addInput("validFeatures")
                .addReader("validMortalityLabels", validMortalityLabels)
                .addOutput("validMortalityLabels")
                .addReader("validLosLabels", validLosLabels)
                .addOutputOneHot("validLosLabels", 0, NB_LOS_TARGETS)
                .addReader("validSurvivalLabels", validSurvivalLabels)
                .addOutputOneHot("validSurvivalLabels", 0, NB_SURVIVAL_TARGETS)
                .build();

        // Test (held out) split, which covers file numbers 3600-3999.
        CSVRecordReader testFeatures = new CSVRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
        CSVRecordReader testMortalityLabels = new CSVRecordReader();
        testMortalityLabels.initialize(new NumberedFileInputSplit(mortalityDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
        CSVRecordReader testLosLabels = new CSVRecordReader();
        testLosLabels.initialize(new NumberedFileInputSplit(losDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
        CSVRecordReader testSurvivalLabels = new CSVRecordReader();
        testSurvivalLabels.initialize(new NumberedFileInputSplit(survivalDir.getAbsolutePath() + "/%d.csv", NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES, NB_TRAIN_EXAMPLES + NB_VALID_EXAMPLES + NB_TEST_EXAMPLES - 1));
        MultiDataSetIterator testData = new RecordReaderMultiDataSetIterator.Builder(BATCH_SIZE)
                .addReader("testFeatures", testFeatures)
                .addInput("testFeatures")
                .addReader("testMortalityLabels", testMortalityLabels)
                .addOutput("testMortalityLabels")
                .addReader("testLosLabels", testLosLabels)
                .addOutputOneHot("testLosLabels", 0, NB_LOS_TARGETS)
                .addReader("testSurvivalLabels", testSurvivalLabels)
                .addOutputOneHot("testSurvivalLabels", 0, NB_SURVIVAL_TARGETS)
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
                .setOutputs("lossMortality", "lossLos", "lossSurvival") // names of outputs (i.e., where loss is compute)
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
                .addLayer("predictLos", new DenseLayer.Builder() // prediction layer
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX)
                                .nOut(NB_LOS_TARGETS)
                                .build(),
                        "dense1")
                .addLayer("lossLos", new LossLayer.Builder(new LossMCXENT()) // loss
                                .build(),
                        "predictLos")
                .addLayer("predictSurvival", new DenseLayer.Builder() // prediction layer
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX)
                                .nOut(NB_SURVIVAL_TARGETS)
                                .build(),
                        "dense1")
                .addLayer("lossSurvival", new LossLayer.Builder(new LossMCXENT()) // loss
                                .build(),
                        "predictSurvival")
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
            ROC rocMortality = new ROC(100);
            ROCMultiClass rocLos = new ROCMultiClass(100);
            ROCMultiClass rocSurvival = new ROCMultiClass(100);
            trainData.reset();
            while (trainData.hasNext()) {
                MultiDataSet batch = trainData.next();
                INDArray[] output = model.output(batch.getFeatures());
                rocMortality.eval(batch.getLabels(0), output[0]);
                rocLos.eval(batch.getLabels(1), output[1]);
                rocSurvival.eval(batch.getLabels(2), output[2]);
            }
            log.info("EPOCH " + epoch + " TRAIN MORTALITY AUC: " + rocMortality.calculateAUC());
            log.info("EPOCH " + epoch + " TRAIN LOS AUC: " + rocLos.calculateAverageAUC());
            log.info("EPOCH " + epoch + " TRAIN SURVIVAL AUC: " + rocSurvival.calculateAverageAUC());

            // loop over batches in validation data to compute validation AUC
            rocMortality = new ROC(100);
            rocLos = new ROCMultiClass(100);
            rocSurvival = new ROCMultiClass(100);
            while (validData.hasNext()) {
                MultiDataSet batch = validData.next();
                INDArray[] output = model.output(batch.getFeatures());
                rocMortality.eval(batch.getLabels(0), output[0]);
                rocLos.eval(batch.getLabels(1), output[1]);
                rocSurvival.eval(batch.getLabels(2), output[2]);
            }
            log.info("EPOCH " + epoch + " VALID MORTALITY AUC: " + rocMortality.calculateAUC());
            log.info("EPOCH " + epoch + " VALID LOS AUC: " + rocLos.calculateAverageAUC());
            log.info("EPOCH " + epoch + " VALID SURVIVAL AUC: " + rocSurvival.calculateAverageAUC());
            trainData.reset();
            validData.reset();
        }

        // Model evaluation: loop over batches in test data to compute test AUC
        ROC rocMortality = new ROC(100);
        ROCMultiClass rocLos = new ROCMultiClass(100);
        ROCMultiClass rocSurvival = new ROCMultiClass(100);
        trainData.reset();
        while (trainData.hasNext()) {
            MultiDataSet batch = trainData.next();
            INDArray[] output = model.output(batch.getFeatures());
            rocMortality.eval(batch.getLabels(0), output[0]);
            rocLos.eval(batch.getLabels(1), output[1]);
            rocSurvival.eval(batch.getLabels(2), output[2]);
        }
        log.info("FINAL TEST MORTALITY AUC: " + rocMortality.calculateAUC());
        log.info("FINAL TEST LOS AUC: " + rocLos.calculateAverageAUC());
        log.info("FINAL TEST SURVIVAL AUC: " + rocSurvival.calculateAverageAUC());
    }
}
