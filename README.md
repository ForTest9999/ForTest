# ForTest

There are the 129 newly-annotated points in ./data/extraLandmarks.mat and extra organ index of each facial curve that used in the experiments of the paper. The extra organ index (extraOrgsInd) is define in ./FCFAN/FCFANtesting.py. You can use this configuration to test the proposed method or generate new annotations (please provide landmarks in extraLandmarks.mat and extraOrgsInd in FCFANtesting.py) on the mean face by yourself.

## Getting started
First, you need to make sure you have installed Python 3.5.2. For that purpose we recommend Anaconda, it has all the necessary libraries except:

    Theano 1.0.4
    Lasagne 0.2.dev1
    OpenCV 3.1.0 or newer
    
The three libraries listed above should be installed for testing the model.

## Download model
The trained model FCFAN.npz that reported in the paper can be downloaded from:
https://www.dropbox.com/s/5kxh43cz5xetlzv/FCFAN.npz?dl=0.
To test the proposed method, you should download this trained model and put it in the current folder.

## Download prepared testing datasets
To test the proposed method, you should download the following datasets and put them in the ./data/ folder.

The prepared common set of 300w public testing set employed in the paper can be downloaded from:
https://www.dropbox.com/s/p2cp3ikxkxjdxpd/commonSet.npz?dl=0.

The prepared challenging set of 300w public testing set employed in the paper can be downloaded from:
https://www.dropbox.com/s/te3klvdj2wz7g2p/challengingSet.npz?dl=0.

The prepared 300w private testing set employed in the paper can be downloaded from:
https://www.dropbox.com/s/l40qmucaiskxt7n/w300Set.npz?dl=0.

## Download original datasets (optional)
If you want to prepare testing data by yourself, you could download 300W, LFPW, HELEN, AFW and IBUG datasets from:
https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/.
then unzip and put them in ./data/images/ folder. After that, run TestSetPreparation.py to prepare testing data , which could take a few minutes.

## Test the model
After all of the steps above, just run FCFANtesting.py to test the model. The resulting annotations are stored in ./300w_Results/ folder.

The parameters you can set in the script are as follows:
* verbose: if True the script will display the error for each image,
* showResults: if True it will show the localized landmarks for each image,
* showCED: if True the Cumulative Error Distribution curve will be shown along with the AUC score,
* normalization: 'centers' for inter-pupil distance, 'corners' for inter-ocular distance, 'diagonal' for bounding box diagonal normalization,
* failureThreshold: the error threshold over which the results are considered to be failures, for inter-ocular distance it should be set to 0.08.

## License
Please note this code and model only allow for the evaluation of the proposed method.

## Contact
If you have any questions or suggestions feel free to contact me at ForTest_cvpr2020@hotmail.com
