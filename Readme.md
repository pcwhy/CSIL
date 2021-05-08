This is the public repository of our paper: Class-Incremental Learning for Wireless Device Identification in IoT, which is available [HERE](Class_incremental_learning_for_device_identification_in_IoT_IoT_16942_2021.pdf)

and [IEEE Internet of Things Journal](https://ieeexplore.ieee.org/document/9425491)

@ARTICLE{9425491,
  author={Liu, Yongxin and Wang, Jian and Li, Jianqiang and Niu, Shuteng and Song, Houbing},
  journal={IEEE Internet of Things Journal}, 
  title={Class-Incremental Learning for Wireless Device Identification in IoT}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JIOT.2021.3078407}}

Our raw dataset is available at https://ieee-dataport.org/documents/ads-b-signals-records-non-cryptographic-identification-and-incremental-learning.
It's a raw ADS-B signal dataset with labels, the dataset is captured using a BladeRF2 SDR receiver @ 1090MHz with a sample rate of 10MHz.

Please goto [IEEE Dataport](https://ieee-dataport.org/documents/ads-b-signals-records-non-cryptographic-identification-and-incremental-learning) for the dataset (adsb_bladerf2_10M_qt0.mat) and preprocessed data (adsb-107loaded.mat).

Sample code for data preprocessing and incremental learning are in [ContinualLearning](https://github.com/pcwhy/CSIL/tree/main/ContinualLearning)

The code that drives the discovery of Remark 1 is in the [numerical simulation folder](https://github.com/pcwhy/CSIL/tree/main/numericalSimOfDoC)

Comparison of various incremental learning algorithms are in [ContinualLearning/WorkStage](https://github.com/pcwhy/CSIL/tree/main/ContinualLearning/WorkStage)
