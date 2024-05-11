import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.UIPerformanceListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class MusicGenreClassifier {

    public static void main(String[] args) throws Exception {

        int height = 28;
        int width = 28;
        int channels = 1;
        int rngSeed = 123;
        int numEpochs = 1;
        int batchSize = 100;

        // Get the DataSetIterators:
        DataSetIterator trainIter = new Cifar10DataSetIterator(batchSize, true, rngSeed);
        DataSetIterator testIter = new Cifar10DataSetIterator(batchSize, false, rngSeed);

        //Configure the neural network:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .list()
            .layer(new ConvolutionLayer.Builder(5, 5)
                .nIn(channels)
                .stride(1, 1)
                .nOut(20)
                .activation(Activation.IDENTITY)
                .weightInit(WeightInit.XAVIER)
                .build())
            .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(new DenseLayer.Builder().activation(Activation.RELU)
                .nOut(500).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build())
            .setInputType(InputType.convolutionalFlat(height, width, channels))
            .backprop(true).pretrain(false).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(10), new PerformanceListener.Builder().build());
        net.setListeners(new UIPerformanceListener());

        System.out.println("Training model...");
        for (int i = 0; i < numEpochs; i++) {
            net.fit(trainIter);
            System.out.println("Completed epoch " + i);
        }

        System.out.println("Evaluating model...");
        Evaluation eval = net.evaluate(testIter);
        System.out.println(eval.stats());

        System.out.println("Saving model...");
        File locationToSave = new File("MusicGenreClassifier.zip");
        net.save(locationToSave, true);

        System.out.println("Model saved at " + locationToSave.getAbsolutePath());
    }
}
