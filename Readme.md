Our raw dataset is available at https://ieee-dataport.org/documents/ads-b-signals-records-non-cryptographic-identification-and-incremental-learning.
It's a raw ADS-B signal dataset with labels, the dataset is captured using a BladeRF2 SDR receiver @ 1090MHz with a sample rate of 10MHz.

The paper is available [HERE](Class_incremental_learning_for_device_identification_in_IoT_IoT_16942_2021.pdf)

Please goto [IEEE Dataport](https://ieee-dataport.org/documents/ads-b-signals-records-non-cryptographic-identification-and-incremental-learning) for the dataset (adsb_bladerf2_10M_qt0.mat) and preprocessed data (adsb-107loaded.mat).

Sample code for data preprocessing and incremental learning are in [ContinualLearning](https://github.com/pcwhy/CSIL/tree/main/ContinualLearning)

The code that drives the discovery of Remark 1 is in the [numerical simulation folder](https://github.com/pcwhy/CSIL/tree/main/numericalSimOfDoC)

Comparison of various incremental learning algorithms are in [ContinualLearning/WorkStage](https://github.com/pcwhy/CSIL/tree/main/ContinualLearning/WorkStage)
