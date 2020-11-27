Implementation of MalJPEG solution for the detection of malicious images , based on A. Cohen, N. Nissim and Y. Elovici, "MalJPEG: Machine Learning Based Solution for the Detection of Malicious JPEG Images," in IEEE Access, vol. 8, pp. 19997-20011, 2020, doi: 10.1109/ACCESS.2020.2969022.
Link to the paper: https://ieeexplore.ieee.org/document/8967109

Dataset: For benign images https://www.kaggle.com/hsankesara/flickr-image-dataset
         For malicious images we used https://virusshare.com/

Language -> Python3.8

We have implemented the main proposal of extracting the MalJPEG features from a JPEG and training them on different ML Models.

On an Ubuntu machine, 

Install the required libraries using:

        pip3 install -r requirements.txt

Run the feature extractor on a directory of images:

        python3 feature_extractor.py /path/to/file type_of_data

Next to create the datasets:

        python3 dataset_creation.py

Finally to train the models, 

        python3 various_models.py
