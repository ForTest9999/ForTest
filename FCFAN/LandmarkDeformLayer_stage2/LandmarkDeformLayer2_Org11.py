from lasagne.layers import MergeLayer
from theano import tensor as T
import theano
import numpy as np

class LandmarkDeformLayer2_Org11(MergeLayer):
    def __init__(self, moms, init_landmarks, n_T, sigmaV=9.3308, **kwargs):
        super(LandmarkDeformLayer2_Org11, self).__init__([moms, init_landmarks], **kwargs)

        self.sigmaV2 = sigmaV**2
        self.n_T = n_T
        self.tau = 1 / (self.n_T - 1)

    def get_output_shape_for(self, input_shape):
        return (None, 136)

    def compute_landmarks_helper(self, moms, init_landmarks):
        moms = T.reshape(moms[:136], (68, 2))  # 68 * 2
        init_landmarks = T.reshape(init_landmarks[:136], (68, 2))

        mask = T.zeros((68, 2))
        mask = T.set_subtensor(mask[60:65, :], np.ones((5, 2)))

        initLandmarks_aftmas = init_landmarks * mask
        moms_aftmas = moms * mask

        dp = T.zeros((68, 2))
        dp1 = T.zeros((68, 2))

        initLandmarks_loca1 = T.alloc(initLandmarks_aftmas[60, :], 68, 2)
        initLandmarks_loca1_aftmas = initLandmarks_loca1 * mask
        initLandmarks_loca2 = T.alloc(initLandmarks_aftmas[61, :], 68, 2)
        initLandmarks_loca2_aftmas = initLandmarks_loca2 * mask
        initLandmarks_loca3 = T.alloc(initLandmarks_aftmas[62, :], 68, 2)
        initLandmarks_loca3_aftmas = initLandmarks_loca3 * mask
        initLandmarks_loca4 = T.alloc(initLandmarks_aftmas[63, :], 68, 2)
        initLandmarks_loca4_aftmas = initLandmarks_loca4 * mask
        initLandmarks_loca5 = T.alloc(initLandmarks_aftmas[64, :], 68, 2)
        initLandmarks_loca5_aftmas = initLandmarks_loca5 * mask

        weight1 = T.zeros((68, 2))
        weight1_val = T.exp(- T.sum((initLandmarks_loca1_aftmas - initLandmarks_aftmas) ** 2, axis=1) / self.sigmaV2)
        weight1 = T.set_subtensor(weight1[:, 0], weight1_val)
        weight1 = T.set_subtensor(weight1[:, 1], weight1_val)
        val1 = T.sum(weight1 * moms_aftmas, axis=0)
        dp = T.set_subtensor(dp[60, :], val1)

        weight2 = T.zeros((68, 2))
        weight2_val = T.exp(- T.sum((initLandmarks_loca2_aftmas - initLandmarks_aftmas) ** 2, axis=1) / self.sigmaV2)
        weight2 = T.set_subtensor(weight2[:, 0], weight2_val)
        weight2 = T.set_subtensor(weight2[:, 1], weight2_val)
        val2 = T.sum(weight2 * moms_aftmas, axis=0)
        dp = T.set_subtensor(dp[61, :], val2)

        weight3 = T.zeros((68, 2))
        weight3_val = T.exp(- T.sum((initLandmarks_loca3_aftmas - initLandmarks_aftmas) ** 2, axis=1) / self.sigmaV2)
        weight3 = T.set_subtensor(weight3[:, 0], weight3_val)
        weight3 = T.set_subtensor(weight3[:, 1], weight3_val)
        val3 = T.sum(weight3 * moms_aftmas, axis=0)
        dp = T.set_subtensor(dp[62, :], val3)

        weight4 = T.zeros((68, 2))
        weight4_val = T.exp(- T.sum((initLandmarks_loca4_aftmas - initLandmarks_aftmas) ** 2, axis=1) / self.sigmaV2)
        weight4 = T.set_subtensor(weight4[:, 0], weight4_val)
        weight4 = T.set_subtensor(weight4[:, 1], weight4_val)
        val4 = T.sum(weight4 * moms_aftmas, axis=0)
        dp = T.set_subtensor(dp[63, :], val4)

        weight5 = T.zeros((68, 2))
        weight5_val = T.exp(- T.sum((initLandmarks_loca5_aftmas - initLandmarks_aftmas) ** 2, axis=1) / self.sigmaV2)
        weight5 = T.set_subtensor(weight5[:, 0], weight5_val)
        weight5 = T.set_subtensor(weight5[:, 1], weight5_val)
        val5 = T.sum(weight5 * moms_aftmas, axis=0)
        dp = T.set_subtensor(dp[64, :], val5)

        deformedShape = initLandmarks_aftmas + (dp * self.tau)

        deformedShape_loca1 = T.alloc(deformedShape[60, :], 68, 2)
        deformedShape_loca2 = T.alloc(deformedShape[61, :], 68, 2)
        deformedShape_loca3 = T.alloc(deformedShape[62, :], 68, 2)
        deformedShape_loca4 = T.alloc(deformedShape[63, :], 68, 2)
        deformedShape_loca5 = T.alloc(deformedShape[64, :], 68, 2)

        weight11 = T.zeros((68, 2))
        weight11_val = T.exp(- T.sum((deformedShape_loca1 - deformedShape) ** 2, axis=1) / self.sigmaV2)
        weight11 = T.set_subtensor(weight11[:, 0], weight11_val)
        weight11 = T.set_subtensor(weight11[:, 1], weight11_val)
        val11 = T.sum(weight11 * moms_aftmas, axis=0)
        dp1 = T.set_subtensor(dp1[60, :], val11)

        weight22 = T.zeros((68, 2))
        weight22_val = T.exp(- T.sum((deformedShape_loca2 - deformedShape) ** 2, axis=1) / self.sigmaV2)
        weight22 = T.set_subtensor(weight22[:, 0], weight22_val)
        weight22 = T.set_subtensor(weight22[:, 1], weight22_val)
        val22 = T.sum(weight22 * moms_aftmas, axis=0)
        dp1 = T.set_subtensor(dp1[61, :], val22)

        weight33 = T.zeros((68, 2))
        weight33_val = T.exp(- T.sum((deformedShape_loca3 - deformedShape) ** 2, axis=1) / self.sigmaV2)
        weight33 = T.set_subtensor(weight33[:, 0], weight33_val)
        weight33 = T.set_subtensor(weight33[:, 1], weight33_val)
        val33 = T.sum(weight33 * moms_aftmas, axis=0)
        dp1 = T.set_subtensor(dp1[62, :], val33)

        weight44 = T.zeros((68, 2))
        weight44_val = T.exp(- T.sum((deformedShape_loca4 - deformedShape) ** 2, axis=1) / self.sigmaV2)
        weight44 = T.set_subtensor(weight44[:, 0], weight44_val)
        weight44 = T.set_subtensor(weight44[:, 1], weight44_val)
        val44 = T.sum(weight44 * moms_aftmas, axis=0)
        dp1 = T.set_subtensor(dp1[63, :], val44)

        weight55 = T.zeros((68, 2))
        weight55_val = T.exp(- T.sum((deformedShape_loca5 - deformedShape) ** 2, axis=1) / self.sigmaV2)
        weight55 = T.set_subtensor(weight55[:, 0], weight55_val)
        weight55 = T.set_subtensor(weight55[:, 1], weight55_val)
        val55 = T.sum(weight55 * moms_aftmas, axis=0)
        dp1 = T.set_subtensor(dp1[64, :], val55)

        output = (deformedShape + dp1 * self.tau).flatten()
        return output

    def get_output_for(self, inputs, **kwargs):

        outputs, updates = theano.scan(self.compute_landmarks_helper, inputs)
        return outputs
