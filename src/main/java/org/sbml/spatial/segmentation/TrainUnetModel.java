package org.sbml.spatial.segmentation;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.zoo.model.UNet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
//import org.deeplearning4j.zoo.ZooModel;

public class TrainUnetModel {

	private static final Logger log = LoggerFactory.getLogger(TrainUnetModel.class);

	public static void main(String[] args) {
		try {
			int batchSize = 10;
			// String home = System.getProperty("user.home");

			String directory = System.getProperty("user.dir");
			String dataPath = directory + File.separator + "dataset";

			String pathToImage;
			if (args.length > 0) {
				pathToImage = args[0];
			} else {
				pathToImage = dataPath + File.separator + "raw_images" + File.separator + "F01_621w1_crop13.tif";
			}

			File rootDir = new File(dataPath + File.separator + "small_dataset");
			PreProcess prep = new PreProcess(rootDir, batchSize);
			DataSetIterator imageDataSetIterator = prep.dataProcessed();

			UIServer uiServer = UIServer.getInstance();

			// StatsStorage statsStorage = new FileStatsStorage(new
			// File("C:\\Users\\Subroto\\StatsLog"));

			// uiServer.attach(statsStorage);

			Map<Integer, Double> learningScheduleMap = new HashMap<>();
			learningScheduleMap.put(0, 0.00005);
			learningScheduleMap.put(200, 0.00001);
			learningScheduleMap.put(600, 0.000005);
			learningScheduleMap.put(800, 0.0000001);
			// learningScheduleMap.put(1000, 0.00001);

			int numEpochs = 1;

			ComputationGraph model = UNet.builder().updater(new Adam(1e-4)).build().init();
			// ComputationGraph model = UNet.builder().updater(new Adam(new
			// MapSchedule(ScheduleType.ITERATION, learningScheduleMap))).build().init();

			// To change the number of input channels from 3 to 1
			// ZooModel unet = UNet.builder().build();
			// unet.setInputShape(new int[][]{{1, 128, 128}});
			// ComputationGraph model = (ComputationGraph) unet.init();

			StatsStorage ss = new InMemoryStatsStorage();
			uiServer.attach(ss);
			model.addListeners(new ScoreIterationListener(), new StatsListener(ss));
			model.fit(imageDataSetIterator, numEpochs);

			log.warn(model.summary());

			Inference infer = new Inference(model, pathToImage, directory);
			infer.imgOut();

		} catch (Exception e) {
			System.err.println("Oooooops");
			e.printStackTrace();
		}
	}
}
