import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

//import org.nd4j.jita.conf.CudaEnvironment;

public class ImportUnetModel {
    private static final Logger log = LoggerFactory.getLogger(ImportUnetModel.class);
    private static final int WIDTH = 256;
    private static final int HEIGHT = 256;
    private static final int CHANNELS = 3;
    private static final String MODEL_PATH = System.getProperty("user.home") + "/unet/unet_membrane.hdf5";
    private static final String IMAGES_PATH = System.getProperty("user.home") + "/unet/data/membrane/train/aug";

    public static void main(String[] args) {
        try {
            int outputNum = 2; // 0 = cell; 1 = limit
            int batchSize = 1;
            // try to load the .h5 model saved from unet
            String pathToImage;
            if (args.length > 0) {
                // we prodived a filename containing model
                pathToImage = args[0];
            } else {
                //pathToImage = System.getProperty("user.home") + "/unet/data/membrane/train/aug/image_0_6144147.png";
                pathToImage = System.getProperty("user.home") + "/unet/data/membrane/test/0.png";
            }
            String pathToModelFile = MODEL_PATH;

            log.info("Using model " + pathToModelFile);

            DataNormalization scaler = new ImagePreProcessingScaler(); // scale image between 0 and 1
            UnetPathLabelGenerator labeler = new UnetPathLabelGenerator();

            File rootDir = new File(System.getProperty("user.home") + "/unet/data/membrane/train/image");
            String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
            Random rng = new Random();
            FileSplit inputSplit = new FileSplit(rootDir,allowedExtensions,rng);
            ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT,WIDTH,CHANNELS,labeler);
            imageRecordReader.initialize(inputSplit);
            int labelIndex = 1;
            DataSetIterator imageDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader,batchSize,labelIndex,labelIndex,true);
            scaler.fit(imageDataSetIterator);
            imageDataSetIterator.setPreProcessor(scaler);

            UIServer uiServer = UIServer.getInstance();

            StatsStorage statsStorage = new FileStatsStorage(new File("stats.log"));

            uiServer.attach(statsStorage);


            Map<Integer, Double> learningScheduleMap = new HashMap<>();
            learningScheduleMap.put(0, 0.00006);
            learningScheduleMap.put(200, 0.00005);
            learningScheduleMap.put(600, 0.0000028);
            learningScheduleMap.put(800, 0.0000060);
            learningScheduleMap.put(1000, 0.000001);

            int seed = 1234;
            int numberOfChannels = 1;
            MultiLayerConfiguration architecture = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .l2(0.0005)
                    .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningScheduleMap)))
                    .weightInit(WeightInit.XAVIER)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Updater.NESTEROVS)
                    .list()
                    // Features detection mask 5 x 5
                    .layer(0, new ConvolutionLayer.Builder(5, 5)
                            .nIn(numberOfChannels)
                            .stride(1, 1)
                            .nOut(10) // Número de kernels (objetos a serem classificados)
                            .activation(Activation.RELU).build())
                    // Pooling layer with MAX Pooling spatial invariance
                    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                    .layer(2, new ConvolutionLayer.Builder(5, 5)
                            .stride(1, 1)
                            .nOut(50)
                            .activation(Activation.RELU).build())
                    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                    .layer(4, new CnnLossLayer.Builder().activation(Activation.SIGMOID).lossFunction(LossFunctions.LossFunction.XENT).build())
                    .setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, CHANNELS))
                    .build();
//            MultiLayerNetwork model = new MultiLayerNetwork(architecture);

            ComputationGraph model  = UNet.builder().updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningScheduleMap))).build().init();
            model.addListeners(new ScoreIterationListener(),new StatsListener(statsStorage));
            model.fit(imageDataSetIterator,10);
//            model.fit(imageDataSetIterator);
            // import model file (model and weights)
//            ComputationGraph model = KerasModelImport.importKerasModelAndWeights(pathToModelFile, true);

            log.warn(model.summary());

            NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
            BufferedImage bufferedBGR = Utils.getBGRBufferedImage(new File(pathToImage));
            INDArray imageNative = loader.asMatrix(bufferedBGR);

            log.warn(imageNative.shapeInfoToString());

            imageNative = imageNative.reshape(new long[]{1, CHANNELS, HEIGHT, WIDTH});
            imageNative = imageNative.divi(255f);

//            INDArray[] output = model.output(imageNative);
//            for (INDArray out : output) {
//                out = out.reshape(CHANNELS, HEIGHT,WIDTH);
//                //out = out.permute(2,1,0);
//                BufferedImage bufferedImage = new BufferedImage(WIDTH,HEIGHT,BufferedImage.TYPE_BYTE_GRAY);
//                for (int i=0; i < WIDTH; i++) {
//                    for (int j=0; j < HEIGHT; j++) {
//                        float f = out.getFloat(new int[]{0,j,i});
//                        int gray = (int)(f*255.0);
////                        if (gray > 127) {
////                            bufferedImage.setRGB(i,j,Color.WHITE.getRGB());
////                        } else {
////                            bufferedImage.setRGB(i,j,Color.BLACK.getRGB());
////                        }
//                      bufferedImage.setRGB(i,j,new Color(gray,gray,gray).getRGB());
//                    }
//                }
//                ImageIO.write(bufferedImage,"png",new File("TestOutputUnet.png"));
//                float[] values = out.toFloatVector();
//                System.out.println(Arrays.toString(values));
//            }
        } catch (Exception e) {
            System.err.println("Oooooops");
            e.printStackTrace();
        }
    }
}

