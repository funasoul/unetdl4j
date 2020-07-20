package package org.sbml.spatial.segmentation;;


import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.common.resources.DL4JResources;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.layers.CnnLossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Adam;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.dataset.api.iterator.MultipleEpochsIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URI;

import javax.imageio.ImageIO;
import javax.swing.*;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.List;

public class LearningKfold {
	
	private static final Logger log = LoggerFactory.getLogger(LearningKfold.class);
    private static final int WIDTH = 128;
    private static final int HEIGHT = 128;
    private static final int CHANNELS = 3;
   
    public static BufferedImage getBGRBufferedImage(BufferedImage bufferedImage) {
        for (int w = 0; w < bufferedImage.getWidth(); w++) {
            for (int h = 0; h < bufferedImage.getHeight(); h++) {
                int p = bufferedImage.getRGB(w, h);
                int a = (p >> 24) & 0xff;
                int r = (p >> 16) & 0xff;
                int g = (p >> 8) & 0xff;
                int b = p & 0xff;
                // swap r (red) and b (blue) channels
                p = (a << 24) | (b << 16) | (g << 8) | r;
                bufferedImage.setRGB(w, h, p);
            }
        }
        return bufferedImage;
    }

    public static BufferedImage getBGRBufferedImage(File imageFile) {
        try {
            BufferedImage bufferedImage = ImageIO.read(imageFile);
            return getBGRBufferedImage(bufferedImage);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

   
	public static void main(String[] args) {
		try {
            int outputNum = 2; // 0 = cell; 1 = limit
            int batchSize = 1;
            String pathToImage;
            if (args.length > 0) {
                // we provided a filename containing model
                pathToImage = args[0];
            } else {
                pathToImage = "C:\\Users\\Subroto\\Desktop\\Cell images\\F01_202w1_crop17.tif";
            }

            DataNormalization scaler = new ImagePreProcessingScaler(); // scale image between 0 and 1
            UNetPathLabelGenerator labeler = new UNetPathLabelGenerator();
            File rootDir = new File("C:\\Users\\Subroto\\Desktop\\small_dataset");
            String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
            Random rng = new Random();
            FileSplit fileSplit = new FileSplit(rootDir,allowedExtensions,rng);
            BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labeler);

            
            int i, k = 4;
            
            double[] folds = new double[k];
            
    		Arrays.fill(folds, 44 / k);

    		InputSplit[] filesInDirSplit = fileSplit.sample(pathFilter, folds);
    		
    		
    		
    		/*for( i = 0; i < k; i++)
    		{
    			URI[] loc = filesInDirSplit[i].locations();
    			System.out.println("Hopefully location for split" + i + "number");
    			System.out.println(loc);
    			long len = filesInDirSplit[i].length();
    			System.out.println("Hopefully size for split" + i + "number");
    			System.out.println(len);
    			filesInDirSplit[i].addNewLocation("C:\\Users\\Subroto\\Desktop\\set");
    		}*/
    		
    		ImageRecordReader[] imageReader = new ImageRecordReader[k];
    		for( i = 0; i < k; i++) {
    			imageReader[i] = new ImageRecordReader(HEIGHT,WIDTH,CHANNELS,labeler);
    			try {
    				imageReader[i].initialize(filesInDirSplit[i]);
    			} 
    			catch (IOException e) {
    				e.printStackTrace();
    			}
    		}

    		RecordReaderDataSetIterator[] set = new RecordReaderDataSetIterator[k];
    		for( i = 0; i < k; i++) {
    			set[i] = new RecordReaderDataSetIterator(imageReader[i], batchSize, 1, 1, true);
    			//File setSaver = new File("C:\\Users\\Subroto\\Desktop\\set["+ i +"]");
    			
    			scaler.fit(set[i]);
                set[i].setPreProcessor(scaler);
    			
    		}
    		int  numEpochs = 10;
            
    		
           /* double splitTrainTest = 0.8;
            
            InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
            InputSplit trainData = inputSplit[0];
            InputSplit testData = inputSplit[1];
            
            ImageRecordReader imageRecordReader = new ImageRecordReader(HEIGHT,WIDTH,CHANNELS,labeler);
            imageRecordReader.initialize(trainData);
            int labelIndex = 1;*/
            
            //KFoldIterator imageDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader,batchSize,labelIndex,labelIndex,true);
            //DataSetIterator imageDataSetIterator = new RecordReaderDataSetIterator(imageRecordReader,batchSize,labelIndex,labelIndex,true);
            //DataSet imageDataSetIterator = (DataSet) new RecordReaderDataSetIterator(imageRecordReader,batchSize,labelIndex,labelIndex,true);
            //DataSet imageData = (DataSet) imageRecordReader;
            //scaler.fit(imageData);
            //imageDataSetIterator.setPreProcessor(scaler);
            
            Map<Integer, Double> learningScheduleMap = new HashMap<>();
            learningScheduleMap.put(0, 0.00005);
            learningScheduleMap.put(200, 0.00001);
            learningScheduleMap.put(600, 0.000005);
            learningScheduleMap.put(800, 0.0000001);
//            learningScheduleMap.put(1000, 0.00001);

            
            
           
            
    		int testFold=0;
            /*model.addListeners(new ScoreIterationListener());*/
            while(testFold<k){
            	System.out.println("Testfold number:" + testFold + "   k value: " + k);
       
            ComputationGraph model  = UNet.builder().updater(new Adam(1e-4)).build().init();
            System.out.println("Initializing new model");
            for(i=0; i<k; i++) {
                if(i==testFold){
                	System.out.println("i number:" + i + "   k value: " + k);
            	    continue;
            	}
                else
                {
                	System.out.println("fitting model");
                	model.addListeners(new ScoreIterationListener());
                	model.fit(set[i], numEpochs);    
               }
                
            } 
            
            //DataSet data = set[testFold].next();
       
            
            Evaluation eval = model.evaluate(set[testFold], set[testFold].getLabels());
    		System.out.println(eval.stats());
            //System.out.println(names);
            //INDArray images = data.getLabels();
            //System.out.println(images);

          //Where to save the model
            File locationTosave = new File("C:\\Users\\Subroto\\Desktop\\unetSave"+ "["+ testFold +"]" + ".zip");
            
            boolean saveUpdater = false;
            
            //ModelSerializer needs Model name, saveUpdater ad Location of saving the model
            
            //ModelSerializer.writeModel(model,locationTosave,saveUpdater);
            testFold++;
            }
            
            
            //Evaluation eval = model.evaluateROC(imageDataSetIterator);
            //log.info(eval.stats(true));
            

            //model.setListeners(new StatsListener(statsStorage));
            
            //log.warn(model.summary());
            
            /*log.info("*****SAVE MODEL******");
            
            //Where to save the model
            File locationTosave = new File("C:\\Users\\Subroto\\Desktop\\unet_save3.zip");
            
            boolean saveUpdater = false;
            
            //ModelSerializer needs Model name, saveUpdater ad Location of saving the model
            
            ModelSerializer.writeModel(model,locationTosave,saveUpdater);
            */

            log.info("*****EVALUATE MODEL******");
            NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
            BufferedImage image = ImageIO.read(new File(pathToImage));
            INDArray imageNative = loader.asMatrix(image);

            log.warn(imageNative.shapeInfoToString());

            imageNative = imageNative.reshape(1, CHANNELS, HEIGHT, WIDTH);
            imageNative = imageNative.divi(255f);

            /*INDArray[] output = model.output(imageNative);
            //INDArray sigmoid = Transforms.sigmoid(output);
            for (INDArray out : output) {
                out = out.reshape(1, HEIGHT,WIDTH);
                //out = out.permute(2,1,0);
                BufferedImage bufferedImage = new BufferedImage(WIDTH,HEIGHT,BufferedImage.TYPE_BYTE_GRAY);
                for ( i=0; i < WIDTH; i++) {
                    for (int j=0; j < HEIGHT; j++) {
                        float f = out.getFloat(new int[]{0,j,i});
                        int gray = (int)(f*255.0);
//                        if (gray > 127) {
//                            bufferedImage.setRGB(i,j,Color.WHITE.getRGB());
//                        } else {
//                            bufferedImage.setRGB(i,j,Color.BLACK.getRGB());
//                        }
                      bufferedImage.setRGB(i,j,new Color(gray,gray,gray).getRGB());
                      /*Color c = new Color(image.getRGB(j, i));
    	              int red = (int)(c.getRed() * 0.299);
    	              int green = (int)(c.getGreen() * 0.587);
    	              int blue = (int)(c.getBlue() *0.114);
    	              Color newColor = new Color(red+green+blue,
    	              red+green+blue,red+green+blue);
    	              bufferedImage.setRGB(j,i,newColor.getRGB());
                    }
                }
                //ImageIO.write(bufferedImage,"png",new File("C:\\Users\\Subroto\\Desktop\\outputUnet20.png"));
//                float[] values = out.toFloatVector();
//                System.out.println(Arrays.toString(values));
            }*/
               
        } catch (Exception e) {
            System.err.println("Oooooops");
            e.printStackTrace();
        }
		
	}

}
