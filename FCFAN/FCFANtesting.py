import numpy as np
import scipy.io as sio
from ImageServer import ImageServer
from FaceAlignment import FaceAlignment
import tests

datasetDir ="../data/"

verbose = True
showResults = True
showCED = True

normalization = 'corners'
failureThreshold = 0.08
n_T = 3

networkFilename = "../FCFAN.npz"
network = FaceAlignment(112, 112, 1, 2, n_T)
network.loadNetwork(networkFilename)

print ("Network being tested: " + networkFilename)
print ("Normalization is set to: " + normalization)
print ("Failure threshold is set to: " + str(failureThreshold))

extraLandmarks = sio.loadmat(datasetDir + 'extraLandmarks.mat')['extraLandmarks']
extraOrgsInd = np.array([[0, 18, 33, 42, 51, 58, 67, 79, 91, 104, 113, 124],
                    [18, 33, 42, 51, 58, 67, 79, 91, 104, 113, 124, 129]])

resultsDir_common = './300w_Results/Common Set/'
resultsDir_challenging = './300w_Results/Challenging Set/'
resultsDir_300w = './300w_Results/300w private set/'

commonSet = ImageServer.Load(datasetDir + "commonSet.npz")
challengingSet = ImageServer.Load(datasetDir + "challengingSet.npz")
w300 = ImageServer.Load(datasetDir + "w300Set.npz")

print ("Processing common subset of the 300W public test set (test sets of LFPW and HELEN)")
commonErrs = tests.LandmarkError(commonSet, network, extraLandmarks, extraOrgsInd,  resultsDir_common, normalization, showResults, verbose)
print ("Processing challenging subset of the 300W public test set (IBUG dataset)")
challengingErrs = tests.LandmarkError(challengingSet, network, extraLandmarks, extraOrgsInd, resultsDir_challenging, normalization, showResults, verbose)

fullsetErrs = commonErrs + challengingErrs
print ("Showing results for the entire 300W pulic test set (IBUG dataset, test sets of LFPW and HELEN")
print("Average error: {0}".format(np.mean(fullsetErrs)))
tests.AUCError(fullsetErrs, failureThreshold, showCurve=showCED)

print ("Processing 300W private test set")
w300Errs = tests.LandmarkError(w300, network, extraLandmarks, extraOrgsInd,  resultsDir_300w, normalization, showResults, verbose)
tests.AUCError(w300Errs, failureThreshold, showCurve=showCED)

