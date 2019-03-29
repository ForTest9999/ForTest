from __future__ import print_function

import lasagne
from lasagne.layers import Conv2DLayer, batch_norm
from lasagne.init import GlorotUniform

import numpy as np
import theano

from scipy import ndimage

import utils
from AffineTransformLayer import AffineTransformLayer
from TransformParamsLayer import TransformParamsLayer
from LandmarkImageLayer import LandmarkImageLayer
from LandmarkTranformLayer import LandmarkTransformLayer
from LandmarkConvergeLayer import LandmarkConvergeLayer
from LandmarkConvergeLayer2 import LandmarkConvergeLayer2
from LandmarkDeformLayer_stage1.LandmarkDeformLayer1_Org1 import LandmarkDeformLayer1_Org1
from LandmarkDeformLayer_stage1.LandmarkDeformLayer1_Org2 import LandmarkDeformLayer1_Org2
from LandmarkDeformLayer_stage1.LandmarkDeformLayer1_Org3 import LandmarkDeformLayer1_Org3
from LandmarkDeformLayer_stage1.LandmarkDeformLayer1_Org4 import LandmarkDeformLayer1_Org4
from LandmarkDeformLayer_stage1.LandmarkDeformLayer1_Org5 import LandmarkDeformLayer1_Org5
from LandmarkDeformLayer_stage1.LandmarkDeformLayer1_Org6 import LandmarkDeformLayer1_Org6
from LandmarkDeformLayer_stage1.LandmarkDeformLayer1_Org7 import LandmarkDeformLayer1_Org7
from LandmarkDeformLayer_stage1.LandmarkDeformLayer1_Org8 import LandmarkDeformLayer1_Org8
from LandmarkDeformLayer_stage1.LandmarkDeformLayer1_Org9 import LandmarkDeformLayer1_Org9
from LandmarkDeformLayer_stage1.LandmarkDeformLayer1_Org10 import LandmarkDeformLayer1_Org10
from LandmarkDeformLayer_stage1.LandmarkDeformLayer1_Org11 import LandmarkDeformLayer1_Org11
from LandmarkDeformLayer_stage1.LandmarkDeformLayer1_Org12 import LandmarkDeformLayer1_Org12

from LandmarkDeformLayer_stage2.LandmarkDeformLayer2_Org1 import LandmarkDeformLayer2_Org1
from LandmarkDeformLayer_stage2.LandmarkDeformLayer2_Org2 import LandmarkDeformLayer2_Org2
from LandmarkDeformLayer_stage2.LandmarkDeformLayer2_Org3 import LandmarkDeformLayer2_Org3
from LandmarkDeformLayer_stage2.LandmarkDeformLayer2_Org4 import LandmarkDeformLayer2_Org4
from LandmarkDeformLayer_stage2.LandmarkDeformLayer2_Org5 import LandmarkDeformLayer2_Org5
from LandmarkDeformLayer_stage2.LandmarkDeformLayer2_Org6 import LandmarkDeformLayer2_Org6
from LandmarkDeformLayer_stage2.LandmarkDeformLayer2_Org7 import LandmarkDeformLayer2_Org7
from LandmarkDeformLayer_stage2.LandmarkDeformLayer2_Org8 import LandmarkDeformLayer2_Org8
from LandmarkDeformLayer_stage2.LandmarkDeformLayer2_Org9 import LandmarkDeformLayer2_Org9
from LandmarkDeformLayer_stage2.LandmarkDeformLayer2_Org10 import LandmarkDeformLayer2_Org10
from LandmarkDeformLayer_stage2.LandmarkDeformLayer2_Org11 import LandmarkDeformLayer2_Org11
from LandmarkDeformLayer_stage2.LandmarkDeformLayer2_Org12 import LandmarkDeformLayer2_Org12


class FaceAlignment(object):
    def __init__(self, height, width, nChannels, nStages, n_T):
        self.landmarkPatchSize = 16

        self.data = theano.tensor.tensor4('inputs', dtype=theano.config.floatX)
        self.targets = theano.tensor.tensor4('targets')

        self.imageHeight = height
        self.imageWidth = width
        self.nChannels = nChannels

        self.errors = []
        self.errorsTrain = []

        self.nStages = nStages

        self.n_T = n_T
        self.OrgInd = np.array([[0, 9, 17, 22, 27, 31, 36, 42, 48, 55, 60, 65],
                                [9, 17, 22, 27, 31, 36, 42, 48, 55, 60, 65, 68]])

    def initializeNetwork(self):
        self.layers = self.createCNN()
        self.network = self.layers['output']
        self.network_moms1 = self.layers['s1_output']
        self.network_output1 = self.layers['s1_landmarks_affine']
        self.network_moms2 = self.layers['s2_output']
        self.transform_params = self.layers['s1_transform_params']

        self.prediction_moms1 = lasagne.layers.get_output(self.network_moms1, deterministic=True)
        self.generate_network_moms1 = theano.function([self.data], [self.prediction_moms1])

        self.prediction_output1 = lasagne.layers.get_output(self.network_output1, deterministic=True)
        self.generate_network_output1 = theano.function([self.data], [self.prediction_output1])

        self.prediction_moms2 = lasagne.layers.get_output(self.network_moms2, deterministic=True)
        self.generate_network_moms2 = theano.function([self.data], [self.prediction_moms2])

        self.prediction_transform_params = lasagne.layers.get_output(self.transform_params, deterministic=True)
        self.generate_network_transform_params = theano.function([self.data], [self.prediction_transform_params])

        self.prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.generate_network_output = theano.function([self.data], [self.prediction])    

    def addDANStage(self, stageIdx, net):
        prevStage = 's' + str(stageIdx - 1)
        curStage = 's' + str(stageIdx)

        # CONNNECTION LAYERS OF PREVIOUS STAGE
        net[prevStage + '_transform_params'] = TransformParamsLayer(net[prevStage + '_landmarks'], self.initLandmarks)
        net[prevStage + '_img_output'] = AffineTransformLayer(net['input'], net[prevStage + '_transform_params'])

        net[prevStage + '_landmarks_affine'] = LandmarkTransformLayer(net[prevStage + '_landmarks'], net[prevStage + '_transform_params'])
        net[prevStage + '_img_landmarks'] = LandmarkImageLayer(net[prevStage + '_landmarks_affine'], (self.imageHeight, self.imageWidth), self.landmarkPatchSize)

        net[prevStage + '_img_feature'] = lasagne.layers.DenseLayer(net[prevStage + '_fc1'], num_units=56 * 56, W=GlorotUniform('relu'))
        net[prevStage + '_img_feature'] = lasagne.layers.ReshapeLayer(net[prevStage + '_img_feature'], (-1, 1, 56, 56))
        net[prevStage + '_img_feature'] = lasagne.layers.Upscale2DLayer(net[prevStage + '_img_feature'], 2)

        # CURRENT STAGE
        net[curStage + '_input'] = batch_norm(lasagne.layers.ConcatLayer(
            [net[prevStage + '_img_output'], net[prevStage + '_img_landmarks'], net[prevStage + '_img_feature']], 1))
        net[curStage + '_conv1_1'] = batch_norm(Conv2DLayer(net[curStage + '_input'], 64, 3, pad='same', W=GlorotUniform('relu')))
        net[curStage + '_conv1_2'] = batch_norm(Conv2DLayer(net[curStage + '_conv1_1'], 64, 3, pad='same', W=GlorotUniform('relu')))
        net[curStage + '_pool1'] = lasagne.layers.Pool2DLayer(net[curStage + '_conv1_2'], 2)

        net[curStage + '_conv2_1'] = batch_norm(Conv2DLayer(net[curStage + '_pool1'], 128, 3, pad=1, W=GlorotUniform('relu')))
        net[curStage + '_conv2_2'] = batch_norm(Conv2DLayer(net[curStage + '_conv2_1'], 128, 3, pad=1, W=GlorotUniform('relu')))
        net[curStage + '_pool2'] = lasagne.layers.Pool2DLayer(net[curStage + '_conv2_2'], 2)

        net[curStage + '_conv3_1'] = batch_norm(Conv2DLayer(net[curStage + '_pool2'], 256, 3, pad=1, W=GlorotUniform('relu')))
        net[curStage + '_conv3_2'] = batch_norm(Conv2DLayer(net[curStage + '_conv3_1'], 256, 3, pad=1, W=GlorotUniform('relu')))
        net[curStage + '_pool3'] = lasagne.layers.Pool2DLayer(net[curStage + '_conv3_2'], 2)

        net[curStage + '_conv4_1'] = batch_norm(Conv2DLayer(net[curStage + '_pool3'], 512, 3, pad=1, W=GlorotUniform('relu')))
        net[curStage + '_conv4_2'] = batch_norm(Conv2DLayer(net[curStage + '_conv4_1'], 512, 3, pad=1, W=GlorotUniform('relu')))
        net[curStage + '_pool4'] = lasagne.layers.Pool2DLayer(net[curStage + '_conv4_2'], 2)

        net[curStage + '_pool4'] = lasagne.layers.FlattenLayer(net[curStage + '_pool4'])
        net[curStage + '_fc1_dropout'] = lasagne.layers.DropoutLayer(net[curStage + '_pool4'], p=0.5)

        net[curStage + '_fc1'] = batch_norm(lasagne.layers.DenseLayer(net[curStage + '_fc1_dropout'], num_units=256, W=GlorotUniform('relu')))

        net[curStage + '_output'] = lasagne.layers.DenseLayer(net[curStage + '_fc1'], num_units=136, nonlinearity=None)

        net[curStage + '_landmarks_Org1'] = LandmarkDeformLayer2_Org1(net[curStage + '_output'], net[prevStage + '_landmarks_affine'], self.n_T)
        net[curStage + '_landmarks_Org2'] = LandmarkDeformLayer2_Org2(net[curStage + '_output'], net[prevStage + '_landmarks_affine'], self.n_T)
        net[curStage + '_landmarks_Org3'] = LandmarkDeformLayer2_Org3(net[curStage + '_output'], net[prevStage + '_landmarks_affine'], self.n_T)
        net[curStage + '_landmarks_Org4'] = LandmarkDeformLayer2_Org4(net[curStage + '_output'], net[prevStage + '_landmarks_affine'], self.n_T)
        net[curStage + '_landmarks_Org5'] = LandmarkDeformLayer2_Org5(net[curStage + '_output'], net[prevStage + '_landmarks_affine'], self.n_T)
        net[curStage + '_landmarks_Org6'] = LandmarkDeformLayer2_Org6(net[curStage + '_output'], net[prevStage + '_landmarks_affine'], self.n_T)
        net[curStage + '_landmarks_Org7'] = LandmarkDeformLayer2_Org7(net[curStage + '_output'], net[prevStage + '_landmarks_affine'], self.n_T)
        net[curStage + '_landmarks_Org8'] = LandmarkDeformLayer2_Org8(net[curStage + '_output'], net[prevStage + '_landmarks_affine'], self.n_T)
        net[curStage + '_landmarks_Org9'] = LandmarkDeformLayer2_Org9(net[curStage + '_output'], net[prevStage + '_landmarks_affine'], self.n_T)
        net[curStage + '_landmarks_Org10'] = LandmarkDeformLayer2_Org10(net[curStage + '_output'], net[prevStage + '_landmarks_affine'], self.n_T)
        net[curStage + '_landmarks_Org11'] = LandmarkDeformLayer2_Org11(net[curStage + '_output'], net[prevStage + '_landmarks_affine'], self.n_T)
        net[curStage + '_landmarks_Org12'] = LandmarkDeformLayer2_Org12(net[curStage + '_output'], net[prevStage + '_landmarks_affine'], self.n_T)

        net[curStage + '_landmarks'] = LandmarkConvergeLayer2(net[curStage + '_landmarks_Org1'], net[curStage + '_landmarks_Org2'],
                                                              net[curStage + '_landmarks_Org3'], net[curStage + '_landmarks_Org4'],
                                                              net[curStage + '_landmarks_Org5'], net[curStage + '_landmarks_Org6'],
                                                              net[curStage + '_landmarks_Org7'], net[curStage + '_landmarks_Org8'],
                                                              net[curStage + '_landmarks_Org9'], net[curStage + '_landmarks_Org10'],
                                                              net[curStage + '_landmarks_Org11'], net[curStage + '_landmarks_Org12'])

        net[curStage + '_landmarks'] = LandmarkTransformLayer(net[curStage + '_landmarks'], net[prevStage + '_transform_params'], True)

    def createCNN(self):
        net = {}
        net['input'] = lasagne.layers.InputLayer(shape=(None, self.nChannels, self.imageHeight, self.imageWidth), input_var=self.data)       
        print("Input shape: {0}".format(net['input'].output_shape))

        #STAGE 1
        net['s1_conv1_1'] = batch_norm(Conv2DLayer(net['input'], 64, 3, pad='same', W=GlorotUniform('relu')))
        net['s1_conv1_2'] = batch_norm(Conv2DLayer(net['s1_conv1_1'], 64, 3, pad='same', W=GlorotUniform('relu')))
        net['s1_pool1'] = lasagne.layers.Pool2DLayer(net['s1_conv1_2'], 2)

        net['s1_conv2_1'] = batch_norm(Conv2DLayer(net['s1_pool1'], 128, 3, pad=1, W=GlorotUniform('relu')))
        net['s1_conv2_2'] = batch_norm(Conv2DLayer(net['s1_conv2_1'], 128, 3, pad=1, W=GlorotUniform('relu')))
        net['s1_pool2'] = lasagne.layers.Pool2DLayer(net['s1_conv2_2'], 2)

        net['s1_conv3_1'] = batch_norm (Conv2DLayer(net['s1_pool2'], 256, 3, pad=1, W=GlorotUniform('relu')))
        net['s1_conv3_2'] = batch_norm (Conv2DLayer(net['s1_conv3_1'], 256, 3, pad=1, W=GlorotUniform('relu')))  
        net['s1_pool3'] = lasagne.layers.Pool2DLayer(net['s1_conv3_2'], 2)
        
        net['s1_conv4_1'] = batch_norm(Conv2DLayer(net['s1_pool3'], 512, 3, pad=1, W=GlorotUniform('relu')))
        net['s1_conv4_2'] = batch_norm (Conv2DLayer(net['s1_conv4_1'], 512, 3, pad=1, W=GlorotUniform('relu')))  
        net['s1_pool4'] = lasagne.layers.Pool2DLayer(net['s1_conv4_2'], 2)
                      
        net['s1_fc1_dropout'] = lasagne.layers.DropoutLayer(net['s1_pool4'], p=0.5)
        net['s1_fc1'] = batch_norm(lasagne.layers.DenseLayer(net['s1_fc1_dropout'], num_units=256, W=GlorotUniform('relu')))

        net['s1_output'] = lasagne.layers.DenseLayer(net['s1_fc1'], num_units=136, nonlinearity=None)

        net['s1_landmarks_Org1'] = LandmarkDeformLayer1_Org1(net['s1_output'], self.initLandmarks, self.n_T)
        net['s1_landmarks_Org2'] = LandmarkDeformLayer1_Org2(net['s1_output'], self.initLandmarks, self.n_T)
        net['s1_landmarks_Org3'] = LandmarkDeformLayer1_Org3(net['s1_output'], self.initLandmarks, self.n_T)
        net['s1_landmarks_Org4'] = LandmarkDeformLayer1_Org4(net['s1_output'], self.initLandmarks, self.n_T)
        net['s1_landmarks_Org5'] = LandmarkDeformLayer1_Org5(net['s1_output'], self.initLandmarks, self.n_T)
        net['s1_landmarks_Org6'] = LandmarkDeformLayer1_Org6(net['s1_output'], self.initLandmarks, self.n_T)
        net['s1_landmarks_Org7'] = LandmarkDeformLayer1_Org7(net['s1_output'], self.initLandmarks, self.n_T)
        net['s1_landmarks_Org8'] = LandmarkDeformLayer1_Org8(net['s1_output'], self.initLandmarks, self.n_T)
        net['s1_landmarks_Org9'] = LandmarkDeformLayer1_Org9(net['s1_output'], self.initLandmarks, self.n_T)
        net['s1_landmarks_Org10'] = LandmarkDeformLayer1_Org10(net['s1_output'], self.initLandmarks, self.n_T)
        net['s1_landmarks_Org11'] = LandmarkDeformLayer1_Org11(net['s1_output'], self.initLandmarks, self.n_T)
        net['s1_landmarks_Org12'] = LandmarkDeformLayer1_Org12(net['s1_output'], self.initLandmarks, self.n_T)

        net['s1_landmarks'] = LandmarkConvergeLayer(net['s1_landmarks_Org1'], net['s1_landmarks_Org2'],
                                                    net['s1_landmarks_Org3'], net['s1_landmarks_Org4'],
                                                    net['s1_landmarks_Org5'], net['s1_landmarks_Org6'],
                                                    net['s1_landmarks_Org7'], net['s1_landmarks_Org8'],
                                                    net['s1_landmarks_Org9'], net['s1_landmarks_Org10'],
                                                    net['s1_landmarks_Org11'], net['s1_landmarks_Org12'])

        for i in range(1, self.nStages):
            self.addDANStage(i + 1, net)

        net['output'] = net['s' + str(self.nStages) + '_landmarks']

        return net

    def loadNetwork(self, filename):
        print('Loading network...')

        with np.load(filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files) - 5)]
            self.errors = f["errors"].tolist()
            self.errorsTrain = f["errorsTrain"].tolist()
            self.meanImg = f["meanImg"]
            self.stdDevImg = f["stdDevImg"]
            self.initLandmarks = f["initLandmarks"]
                
        self.initializeNetwork()
        nParams = len(lasagne.layers.get_all_param_values(self.network))
        lasagne.layers.set_all_param_values(self.network, param_values[:nParams])
        
    def processImg(self, img, inputLandmarks):
        inputImg, transform = self.CropResizeRotate(img, inputLandmarks)
        inputImg = inputImg - self.meanImg
        inputImg = inputImg / self.stdDevImg

        output = self.generate_network_output([inputImg])[0][0]

        landmarks = output.reshape((-1, 2))
        return np.dot(landmarks - transform[1], np.linalg.inv(transform[0]))

    def flow(self, moms, ref, newShape, sigmaV2):
        dp = np.zeros((newShape.shape))
        for i in range(newShape.shape[0]):
            for j in range(ref.shape[0]):
                argin = -((ref[j, 0] - newShape[i, 0])**2 + (ref[j, 1] - newShape[i, 1])**2)/sigmaV2
                argout = np.exp(argin)
                dp[i, 0] = dp[i, 0] + argout * moms[j, 0]
                dp[i, 1] = dp[i, 1] + argout * moms[j, 1]

        newShape_temp = newShape + 0.5 * dp

        dp_ref = np.zeros((ref.shape))
        for i in range(ref.shape[0]):
            for j in range(ref.shape[0]):
                argin = -((ref[j, 0] - ref[i, 0])**2 + (ref[j, 1] - ref[i, 1])**2) / sigmaV2
                argout = np.exp(argin)
                dp_ref[i, 0] = dp_ref[i, 0] + argout * moms[j, 0]
                dp_ref[i, 1] = dp_ref[i, 1] + argout * moms[j, 1]

        ref_temp = ref + 0.5 * dp_ref

        dp1 = np.zeros((newShape.shape))
        for i in range(newShape_temp.shape[0]):
            for j in range(ref.shape[0]):
                argin = -((ref_temp[j, 0] - newShape_temp[i, 0])**2 + (ref_temp[j, 1] - newShape_temp[i, 1])**2)/sigmaV2
                argout = np.exp(argin)
                dp1[i, 0] = dp1[i, 0] + argout * moms[j, 0]
                dp1[i, 1] = dp1[i, 1] + argout * moms[j, 1]

        newShape_final = newShape_temp + 0.5 * dp1

        return newShape_final

    def processImgWithmoms(self, img, inputLandmarks, extraLandmarks, OrgsInd):
        # stage 1
        inputImg, transform = self.CropResizeRotate(img, inputLandmarks)
        inputImg = inputImg - self.meanImg
        inputImg = inputImg / self.stdDevImg

        output_moms1 = self.generate_network_moms1([inputImg])[0][0]
        moms = output_moms1.reshape((-1, 2))

        # Org1
        moms_Org1 = moms[self.OrgInd[0, 0]:self.OrgInd[1, 0], :]
        ref_Org1 = inputLandmarks[self.OrgInd[0, 0]:self.OrgInd[1, 0], :]
        newShape_Org1 = extraLandmarks[OrgsInd[0, 0]:OrgsInd[1, 0], :]
        sigmaV2_Org1 = 36.1889**2
        newShapeDeformed_Org1 = self.flow(moms_Org1, ref_Org1, newShape_Org1, sigmaV2_Org1)

        # Org2
        moms_Org2 = moms[self.OrgInd[0, 1]:self.OrgInd[1, 1], :]
        ref_Org2 = inputLandmarks[self.OrgInd[0, 1]:self.OrgInd[1, 1], :]
        newShape_Org2 = extraLandmarks[OrgsInd[0, 1]:OrgsInd[1, 1], :]
        sigmaV2_Org2 = 32.2242**2
        newShapeDeformed_Org2 = self.flow(moms_Org2, ref_Org2, newShape_Org2, sigmaV2_Org2)

        # Org3
        moms_Org3 = moms[self.OrgInd[0, 2]:self.OrgInd[1, 2], :]
        ref_Org3 = inputLandmarks[self.OrgInd[0, 2]:self.OrgInd[1, 2], :]
        newShape_Org3 = extraLandmarks[OrgsInd[0, 2]:OrgsInd[1, 2], :]
        sigmaV2_Org3 = 11.2157**2
        newShapeDeformed_Org3 = self.flow(moms_Org3, ref_Org3, newShape_Org3, sigmaV2_Org3)

        # Org4
        moms_Org4 = moms[self.OrgInd[0, 3]:self.OrgInd[1, 3], :]
        ref_Org4 = inputLandmarks[self.OrgInd[0, 3]:self.OrgInd[1, 3], :]
        newShape_Org4 = extraLandmarks[OrgsInd[0, 3]:OrgsInd[1, 3], :]
        sigmaV2_Org4 = 11.2157**2
        newShapeDeformed_Org4 = self.flow(moms_Org4, ref_Org4, newShape_Org4, sigmaV2_Org4)

        # Org5
        moms_Org5 = moms[self.OrgInd[0, 4]:self.OrgInd[1, 4], :]
        ref_Org5 = inputLandmarks[self.OrgInd[0, 4]:self.OrgInd[1, 4], :]
        newShape_Org5 = extraLandmarks[OrgsInd[0, 4]:OrgsInd[1, 4], :]
        sigmaV2_Org5 = 7.2351**2
        newShapeDeformed_Org5 = self.flow(moms_Org5, ref_Org5, newShape_Org5, sigmaV2_Org5)

        # Org6
        moms_Org6 = moms[self.OrgInd[0, 5]:self.OrgInd[1, 5], :]
        ref_Org6 = inputLandmarks[self.OrgInd[0, 5]:self.OrgInd[1, 5], :]
        newShape_Org6 = extraLandmarks[OrgsInd[0, 5]:OrgsInd[1, 5], :]
        sigmaV2_Org6 = 6.5614**2
        newShapeDeformed_Org6 = self.flow(moms_Org6, ref_Org6, newShape_Org6, sigmaV2_Org6)

        # Org7
        moms_Org7 = moms[self.OrgInd[0, 6]:self.OrgInd[1, 6], :]
        ref_Org7 = inputLandmarks[self.OrgInd[0, 6]:self.OrgInd[1, 6], :]
        newShape_Org7 = extraLandmarks[OrgsInd[0, 6]:OrgsInd[1, 6], :]
        sigmaV2_Org7 = 6.4939**2
        newShapeDeformed_Org7 = self.flow(moms_Org7, ref_Org7, newShape_Org7, sigmaV2_Org7)

        # Org8
        moms_Org8 = moms[self.OrgInd[0, 7]:self.OrgInd[1, 7], :]
        ref_Org8 = inputLandmarks[self.OrgInd[0, 7]:self.OrgInd[1, 7], :]
        newShape_Org8 = extraLandmarks[OrgsInd[0, 7]:OrgsInd[1, 7], :]
        sigmaV2_Org8 = 6.4939**2
        newShapeDeformed_Org8 = self.flow(moms_Org8, ref_Org8, newShape_Org8, sigmaV2_Org8)

        # Org9
        moms_Org9 = moms[self.OrgInd[0, 8]:self.OrgInd[1, 8], :]
        ref_Org9 = inputLandmarks[self.OrgInd[0, 8]:self.OrgInd[1, 8], :]
        newShape_Org9 = extraLandmarks[OrgsInd[0, 8]:OrgsInd[1, 8], :]
        sigmaV2_Org9 = 12.0824**2
        newShapeDeformed_Org9 = self.flow(moms_Org9, ref_Org9, newShape_Org9, sigmaV2_Org9)

        # Org10
        moms_Org10 = moms[self.OrgInd[0, 9]:self.OrgInd[1, 9], :]
        ref_Org10 = inputLandmarks[self.OrgInd[0, 9]:self.OrgInd[1, 9], :]
        newShape_Org10 = extraLandmarks[OrgsInd[0, 9]:OrgsInd[1, 9], :]
        sigmaV2_Org10 = 7.9841**2
        newShapeDeformed_Org10 = self.flow(moms_Org10, ref_Org10, newShape_Org10, sigmaV2_Org10)

        # Org11
        moms_Org11 = moms[self.OrgInd[0, 10]:self.OrgInd[1, 10], :]
        ref_Org11 = inputLandmarks[self.OrgInd[0, 10]:self.OrgInd[1, 10], :]
        newShape_Org11 = extraLandmarks[OrgsInd[0, 10]:OrgsInd[1, 10], :]
        sigmaV2_Org11 = 9.3308**2
        newShapeDeformed_Org11 = self.flow(moms_Org11, ref_Org11, newShape_Org11, sigmaV2_Org11)

        # Org12
        moms_Org12 = moms[self.OrgInd[0, 11]:self.OrgInd[1, 11], :]
        ref_Org12 = inputLandmarks[self.OrgInd[0, 11]:self.OrgInd[1, 11], :]
        newShape_Org12 = extraLandmarks[OrgsInd[0, 11]:OrgsInd[1, 11], :]
        sigmaV2_Org12 = 3.0845**2
        newShapeDeformed_Org12 = self.flow(moms_Org12, ref_Org12, newShape_Org12, sigmaV2_Org12)

        newShapeDeformed = np.vstack((newShapeDeformed_Org1, newShapeDeformed_Org2, newShapeDeformed_Org3, newShapeDeformed_Org4,
                                      newShapeDeformed_Org5, newShapeDeformed_Org6, newShapeDeformed_Org7, newShapeDeformed_Org8,
                                      newShapeDeformed_Org9, newShapeDeformed_Org10, newShapeDeformed_Org11, newShapeDeformed_Org12))

        # stage 2
        output_ShapeDeformed_trans = self.generate_network_output1([inputImg])[0][0]
        ShapeDeformed_trans = output_ShapeDeformed_trans.reshape((-1, 2))
        output_moms2 = self.generate_network_moms2([inputImg])[0][0]
        moms2 = output_moms2.reshape((-1, 2))

        output_transParams = self.generate_network_transform_params([inputImg])[0][0]
        A = np.zeros((2, 2))
        A[0, 0] = output_transParams[0]
        A[0, 1] = output_transParams[1]
        A[1, 0] = output_transParams[2]
        A[1, 1] = output_transParams[3]
        t = output_transParams[4:6]
        A_inv = np.linalg.inv(A)
        t_inv = np.dot(-t, A_inv)

        newShapeDeformed_trans = np.dot(newShapeDeformed, A) + t

        # Org1
        moms2_Org1 = moms2[self.OrgInd[0, 0]:self.OrgInd[1, 0], :]
        ref2_Org1 = ShapeDeformed_trans[self.OrgInd[0, 0]:self.OrgInd[1, 0], :]
        newShape2_Org1 = newShapeDeformed_trans[OrgsInd[0, 0]:OrgsInd[1, 0], :]
        newShape2Deformed_Org1 = self.flow(moms2_Org1, ref2_Org1, newShape2_Org1, sigmaV2_Org1)

        # Org2
        moms2_Org2 = moms2[self.OrgInd[0, 1]:self.OrgInd[1, 1], :]
        ref2_Org2 = ShapeDeformed_trans[self.OrgInd[0, 1]:self.OrgInd[1, 1], :]
        newShape2_Org2 = newShapeDeformed_trans[OrgsInd[0, 1]:OrgsInd[1, 1], :]
        newShape2Deformed_Org2 = self.flow(moms2_Org2, ref2_Org2, newShape2_Org2, sigmaV2_Org2)

        # Org3
        moms2_Org3 = moms2[self.OrgInd[0, 2]:self.OrgInd[1, 2], :]
        ref2_Org3 = ShapeDeformed_trans[self.OrgInd[0, 2]:self.OrgInd[1, 2], :]
        newShape2_Org3 = newShapeDeformed_trans[OrgsInd[0, 2]:OrgsInd[1, 2], :]
        newShape2Deformed_Org3 = self.flow(moms2_Org3, ref2_Org3, newShape2_Org3, sigmaV2_Org3)

        # Org4
        moms2_Org4 = moms2[self.OrgInd[0, 3]:self.OrgInd[1, 3], :]
        ref2_Org4 = ShapeDeformed_trans[self.OrgInd[0, 3]:self.OrgInd[1, 3], :]
        newShape2_Org4 = newShapeDeformed_trans[OrgsInd[0, 3]:OrgsInd[1, 3], :]
        newShape2Deformed_Org4 = self.flow(moms2_Org4, ref2_Org4, newShape2_Org4, sigmaV2_Org4)

        # Org5
        moms2_Org5 = moms2[self.OrgInd[0, 4]:self.OrgInd[1, 4], :]
        ref2_Org5 = ShapeDeformed_trans[self.OrgInd[0, 4]:self.OrgInd[1, 4], :]
        newShape2_Org5 = newShapeDeformed_trans[OrgsInd[0, 4]:OrgsInd[1, 4], :]
        newShape2Deformed_Org5 = self.flow(moms2_Org5, ref2_Org5, newShape2_Org5, sigmaV2_Org5)

        # Org6
        moms2_Org6 = moms2[self.OrgInd[0, 5]:self.OrgInd[1, 5], :]
        ref2_Org6 = ShapeDeformed_trans[self.OrgInd[0, 5]:self.OrgInd[1, 5], :]
        newShape2_Org6 = newShapeDeformed_trans[OrgsInd[0, 5]:OrgsInd[1, 5], :]
        newShape2Deformed_Org6 = self.flow(moms2_Org6, ref2_Org6, newShape2_Org6, sigmaV2_Org6)

        # Org7
        moms2_Org7 = moms2[self.OrgInd[0, 6]:self.OrgInd[1, 6], :]
        ref2_Org7 = ShapeDeformed_trans[self.OrgInd[0, 6]:self.OrgInd[1, 6], :]
        newShape2_Org7 = newShapeDeformed_trans[OrgsInd[0, 6]:OrgsInd[1, 6], :]
        newShape2Deformed_Org7 = self.flow(moms2_Org7, ref2_Org7, newShape2_Org7, sigmaV2_Org7)

        # Org8
        moms2_Org8 = moms2[self.OrgInd[0, 7]:self.OrgInd[1, 7], :]
        ref2_Org8 = ShapeDeformed_trans[self.OrgInd[0, 7]:self.OrgInd[1, 7], :]
        newShape2_Org8 = newShapeDeformed_trans[OrgsInd[0, 7]:OrgsInd[1, 7], :]
        newShape2Deformed_Org8 = self.flow(moms2_Org8, ref2_Org8, newShape2_Org8, sigmaV2_Org8)

        # Org9
        moms2_Org9 = moms2[self.OrgInd[0, 8]:self.OrgInd[1, 8], :]
        ref2_Org9 = ShapeDeformed_trans[self.OrgInd[0, 8]:self.OrgInd[1, 8], :]
        newShape2_Org9 = newShapeDeformed_trans[OrgsInd[0, 8]:OrgsInd[1, 8], :]
        newShape2Deformed_Org9 = self.flow(moms2_Org9, ref2_Org9, newShape2_Org9, sigmaV2_Org9)

        # Org10
        moms2_Org10 = moms2[self.OrgInd[0, 9]:self.OrgInd[1, 9], :]
        ref2_Org10 = ShapeDeformed_trans[self.OrgInd[0, 9]:self.OrgInd[1, 9], :]
        newShape2_Org10 = newShapeDeformed_trans[OrgsInd[0, 9]:OrgsInd[1, 9], :]
        newShape2Deformed_Org10 = self.flow(moms2_Org10, ref2_Org10, newShape2_Org10, sigmaV2_Org10)

        # Org11
        moms2_Org11 = moms2[self.OrgInd[0, 10]:self.OrgInd[1, 10], :]
        ref2_Org11 = ShapeDeformed_trans[self.OrgInd[0, 10]:self.OrgInd[1, 10], :]
        newShape2_Org11 = newShapeDeformed_trans[OrgsInd[0, 10]:OrgsInd[1, 10], :]
        newShape2Deformed_Org11 = self.flow(moms2_Org11, ref2_Org11, newShape2_Org11, sigmaV2_Org11)

        # Org12
        moms2_Org12 = moms2[self.OrgInd[0, 11]:self.OrgInd[1, 11], :]
        ref2_Org12 = ShapeDeformed_trans[self.OrgInd[0, 11]:self.OrgInd[1, 11], :]
        newShape2_Org12 = newShapeDeformed_trans[OrgsInd[0, 11]:OrgsInd[1, 11], :]
        newShape2Deformed_Org12 = self.flow(moms2_Org12, ref2_Org12, newShape2_Org12, sigmaV2_Org12)

        newShape2Deformed = np.vstack((newShape2Deformed_Org1, newShape2Deformed_Org2, newShape2Deformed_Org3, newShape2Deformed_Org4,
                                       newShape2Deformed_Org5, newShape2Deformed_Org6, newShape2Deformed_Org7, newShape2Deformed_Org8,
                                       newShape2Deformed_Org9, newShape2Deformed_Org10, newShape2Deformed_Org11, newShape2Deformed_Org12))

        newShape2Deformed_trans = np.dot(newShape2Deformed, A_inv) + t_inv

        return newShape2Deformed_trans

    def processNormalizedImg(self, img):
        inputImg = img.astype(np.float32)
        inputImg = inputImg - self.meanImg
        inputImg = inputImg / self.stdDevImg

        output = self.generate_network_output([inputImg])[0][0]

        landmarks = output.reshape((-1, 2))
        return landmarks

    def CropResizeRotate(self, img, inputShape):
        A, t = utils.bestFit(self.initLandmarks, inputShape, True)
    
        A2 = np.linalg.inv(A)
        t2 = np.dot(-t, A2)

        outImg = np.zeros((self.nChannels, self.imageHeight, self.imageWidth), dtype=np.float32)
        for i in range(img.shape[0]):
            outImg[i] = ndimage.interpolation.affine_transform(img[i], A2, t2[[1, 0]], output_shape=(self.imageHeight, self.imageWidth))

        return outImg, [A, t]
