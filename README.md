Implementation of the MalJPEG paper published in IEEE Xplore by Avaid Cohen et al. in January 2020.

Language -> Python

We have implemented the main proposal of extracting the MalJPEG features from a JPEG and training them on different ML Models.

The malicious images were obtained from virusshare.com, and the benign images from a Kaggle dataset.

Install the required libraries using:

        pip3 install -r requirements.txt

Run the feature extractor on a directory of images:

        python3 feature_extractor.py /path/to/file

Next to create the datasets:

        python3 dataset_creation.py

Finally to train the models, 

        python3 various_models.py
