# object-reid

## Dataset
* Download 3RScan -> https://github.com/WaldJohannaU/3RScan
* Render the 3RScan dataset with the Rio_Renderer -> https://github.com/WaldJohannaU/3RScan/tree/master/c%2B%2B/rio_renderer
* Run the FrameFilter component to generate the 2DInstances.txt file
 for more details on how to prepare teh dataset : https://github.com/lukasHoel/3rscan-triplet-dataset-toolkit (How to get started)
## ORB
* **orb-premiers-pas.ipynb** (first hands on with ORB on a custon dataset and realtime video)
* **orb-3rscan.ipynb** (testing ORB on the 3RScan dataset, make sure to indicate the datapath at utils_orb.py (data not included))
## Deep learning
make sure to indicate the datapath at utils_training.py & utils_test.py (data not included)
* **dl-triplets.ipynb** (triplet geenration &dataloader creation)
* **dl-training.ipynb** (training & saving the models, all the models are included in models.py)
* **dl-testing.ipynb** (testing the trained models: clustering + metrics)
* **dl-testing-pretrained.ipynb** (testing pretrained models: clustering + metrics)