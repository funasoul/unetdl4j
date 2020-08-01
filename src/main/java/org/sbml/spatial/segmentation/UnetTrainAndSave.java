package org.sbml.spatial.segmentation;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

//This class is for training the UNet model on any number of images and then saving it.
public class UnetTrainAndSave {

	private static final Logger log = LoggerFactory.getLogger(UnetTrainAndSave.class);

	public static void main(String[] args) {
		try {
			int batchSize = 10;
			// String home = System.getProperty("user.home");
			String directory = System.getProperty("user.dir");
			String dataPath = directory + File.separator + "dataset";

			File rootDir = new File(dataPath + File.separator + "small_dataset");
			PreProcess prep = new PreProcess(rootDir, batchSize);
			DataSetIterator imageDataSetIterator = prep.dataProcessed();

			Map<Integer, Double> learningScheduleMap = new HashMap<>();
			learningScheduleMap.put(0, 0.00005);
			learningScheduleMap.put(200, 0.00001);
			learningScheduleMap.put(600, 0.000005);
			learningScheduleMap.put(800, 0.0000001);
//	            learningScheduleMap.put(1000, 0.00001);

			int numEpochs = 100;

			ComputationGraph model = UNet.builder().updater(new Adam(1e-4)).build().init();
//	            ComputationGraph model  = UNet.builder().updater(new Adam(new MapSchedule(ScheduleType.ITERATION, learningScheduleMap))).build().init();
			model.addListeners(new ScoreIterationListener());
			model.fit(imageDataSetIterator, numEpochs);

			// log.warn(model.summary());

			log.info("*****SAVE MODEL******");

			// Location for saving the model
			// File locationTosave = new File(home + File.separator + "unetSave.zip");
			File locationTosave = new File(directory + File.separator + "unetSave.zip");
			boolean saveUpdater = false;
			// ModelSerializer needs Model name, Location of saving the model and
			// saveUpdater.
			ModelSerializer.writeModel(model, locationTosave, saveUpdater);

		} catch (Exception e) {
			System.err.println("Oooooops");
			e.printStackTrace();
		}
	}
}
