package org.sbml.spatial.segmentation;


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
import java.util.Random;
import javax.imageio.ImageIO;
import javax.swing.*;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class crossVal {
	
	private static final Logger log = LoggerFactory.getLogger(crossVal.class);
    private static final int WIDTH = 128;
    private static final int HEIGHT = 128;
    private static final int CHANNELS = 3;

   
	public static void main(String[] args) {
		try {
            int outputNum = 2; 
            int batchSize = 1;
            
            DataNormalization scaler = new ImagePreProcessingScaler(); // scale image between 0 and 1
            UnetPathLabelGenerator labeler = new UnetPathLabelGenerator();
            File rootDir = new File("C:\\Users\\Subroto\\Desktop\\small_dataset");
            String[] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
            Random rng = new Random();
            FileSplit fileSplit = new FileSplit(rootDir,allowedExtensions,rng);
            BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labeler);

            //Since this dataset had 44 cell images and 44 corresponding grounth truth images, we can split for 4-fold cross validation (k=4 )
            int i, k = 4;
            double[] folds = new double[k];
            Arrays.fill(folds, 44 / k);     //setting weights for splitting the input file 
            InputSplit[] filesInDirSplit = fileSplit.sample(pathFilter, folds);
    		
    		
    		/*Tried to obtained location of the split files ----> Did not know how to interpret the results
        for( i = 0; i < k; i++)
    		{
    			URI[] loc = filesInDirSplit[i].locations();
    			System.out.println("Hopefully location for split " + i + " number");
    			System.out.println(loc);
    			long len = filesInDirSplit[i].length();
    			System.out.println("Hopefully size for split" + i + "number");
    			System.out.println(len);   //This gave correct length
    			File 
    		}*/
    		
        //ImageRecordReader for each split input
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

        //RecordReaderDatasetIterator for each split input
     		RecordReaderDataSetIterator[] set = new RecordReaderDataSetIterator[k];
    		for( i = 0; i < k; i++) {
    			set[i] = new RecordReaderDataSetIterator(imageReader[i], batchSize, 1, 1, true);
    			scaler.fit(set[i]);
          set[i].setPreProcessor(scaler);
    			
    		}
    		int  numEpochs = 10;
            
            Map<Integer, Double> learningScheduleMap = new HashMap<>();
            learningScheduleMap.put(0, 0.00005);
            learningScheduleMap.put(200, 0.00001);
            learningScheduleMap.put(600, 0.000005);
            learningScheduleMap.put(800, 0.0000001);
//            learningScheduleMap.put(1000, 0.00001);

        
        //Training the model for cross validation
    		int testFold=0;
            while(testFold<k){
            	//System.out.println("Testfold number:" + testFold + "   k value: " + k);
       
            ComputationGraph model  = UNet.builder().updater(new Adam(1e-4)).build().init(); //Initializing a new model 
            //System.out.println("Initializing new model");
            for(i=0; i<k; i++) {
                if(i==testFold){
                	//System.out.println("i number:" + i + "   testFold value: " +  testFold);
            	    continue; //Model is not trained on the testfold
            	}
                else
                {
                	System.out.println("fitting model");  //Model gets trained on all folds except the testFold
                	//model.addListeners(new ScoreIterationListener());
                	//model.fit(set[i], numEpochs);    
               }
                
            } 
               

            //Now saving the model weights so that it can be tested later with the corresponding testfold. Need a method to save the testfold so that the images and labels can be separated. 
            File locationTosave = new File("C:\\Users\\Subroto\\Desktop\\unetSave"+ "["+ testFold +"]" + ".zip"); //So that I know which testFold to test this model against
            boolean saveUpdater = false;
            ModelSerializer.writeModel(model,locationTosave,saveUpdater);
            testFold++;
            }
           
               
        } catch (Exception e) {
            System.err.println("Oooooops");
            e.printStackTrace();
        }
		
	}

}
