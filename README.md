# ForTest

There are the 129 newly-annotated points in ./data/extraLandmarks.mat and Organ index of each facial curve that used in the experiments of the paper, Organ index is define in ./FCFAN/FCFANtesting.py. You can use this configuration to test the proposed method or generate new annotations (provide landmarks in extraLandmarks.mat and Organ index in FCFANtesting.py) on the mean face by yourself.

First, you need to make sure you have installed Python 3.5.2. For that purpose we recommend Anaconda, it has all the necessary libraries except:

    Theano 1.0.4
    Lasagne 0.2.dev1
    OpenCV 3.1.0 or newer
    
The three libraries listed above should be installed for testing the model.

## The trained model FCFAN.npz in the paper can be downloaded from this url:
https://www.dropbox.com/s/5kxh43cz5xetlzv/FCFAN.npz?dl=0
To test the proposed method, you should download this trained model and put it in the current folder.

## To test the proposed method, you should download the following datasets and put them in the ./data folder.
The prepared testing data set common set of 300w public set in the paper can be downloaded from this url:
https://www.dropbox.com/s/p2cp3ikxkxjdxpd/commonSet.npz?dl=0

The prepared testing data set challenging set of 300w public set in the paper can be downloaded from this url:
https://www.dropbox.com/s/te3klvdj2wz7g2p/challengingSet.npz?dl=0

The prepared testing data set 300w private set in the paper can be downloaded from this url:
https://www.dropbox.com/s/l40qmucaiskxt7n/w300Set.npz?dl=0

## If you want to prepare testing data by yourself, you could download 300W, LFPW, HELEN, AFW and IBUG datasets form this url:
https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
then unzip and put them in ./data/images/ folder. 

After all the steps above, just run FCFANtesting.py. 

The parameters you can set in the script are as follows:
* verbose: if True the script will display the error for each image,
* showResults: if True it will show the localized landmarks for each image,
* showCED: if True the Cumulative Error Distribution curve will be shown along with the AUC score,
* normalization: 'centers' for inter-pupil distance, 'corners' for inter-ocular distance, 'diagonal' for bounding box diagonal normalization,
* failureThreshold: the error threshold over which the results are considered to be failures, for inter-ocular distance it should be set to 0.08.

## License
Please note this code and model only could used for evaluation of the paper submitted to the conference.

## Contact
If you have any questions or suggestions feel free to contact me at fortest_iccv2019@hotmail.com
