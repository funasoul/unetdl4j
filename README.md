# unetdl4j
Training Unet with DL4J

Basic use of DL4J for training a Unet network.

Use of zhixuhao Github ((https://github.com/zhixuhao/unet)) example for data/images.

## How to compile
```sh
% mvn compile
% mvn package   # create jar
```

## How to run
```sh
% java -cp target/segmentation-1.0-SNAPSHOT-bin.jar org.sbml.spatial.segmentation.TrainUnetModel
```
## Additional Datasets

### For Training

- 100 microscopic cellular images with the corresponding labels: https://drive.google.com/drive/folders/1u3SgJYb1LObpboEKkURQr3Mh7FrPrf_8?usp=sharing

- 300 microscopic cellular images with the corresponding labels: https://drive.google.com/drive/folders/1UUq6W-3P7Mg-eSE6_UJSCQaC8Xazc3zH?usp=sharing

- 44 microscopic cellular images with the corresponding labels (easy training): https://drive.google.com/drive/folders/1Ox0fi1V9dwBXPHisgLc9kjaIfZFZ27dy?usp=sharing

 - 44 microscopic cellular images with the corresponding labels (split for cross validation): https://drive.google.com/drive/folders/1eyRLg1s110ID-T8Oa4Je0xB9fqyy-zZr?usp=sharing


### For Testing

 - 20 microscopic cellular images: https://drive.google.com/drive/folders/1lNphWDWUDq6U4K25kP-zHL8U4ETawuDE?usp=sharing
