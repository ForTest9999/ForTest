from lasagne.layers import Layer
from theano import tensor as T
import theano
import numpy as np

class LandmarkDeformLayer1_Org12(Layer):
    def __init__(self, moms, init_landmarks, n_T, sigmaV=3.0845, **kwargs):
        super(LandmarkDeformLayer1_Org12, self).__init__(moms, **kwargs)

        self.init_landmarks = init_landmarks
        self.sigmaV2 = sigmaV**2
        self.n_T = n_T
        self.tau = 1 / (self.n_T - 1)

    def get_output_shape_for(self, input_shape):
        return (None, 136)


    def compute_landmarks_helper(self, moms):
        moms = T.reshape(moms[:136], (68, 2))  # 68 * 2

        mask = T.zeros((68, 2))
        mask = T.set_subtensor(mask[65:68, :], np.ones((3, 2)))

        initLandmarks_aftmas = self.init_landmarks * mask
        moms_aftmas = moms * mask

        dp = T.zeros((68, 2))
        dp1 = T.zeros((68, 2))

        initLandmarks_loca1 = T.alloc(initLandmarks_aftmas[65, :], 68, 2)
        initLandmarks_loca1_aftmas = initLandmarks_loca1 * mask
        initLandmarks_loca2 = T.alloc(initLandmarks_aftmas[66, :], 68, 2)
        initLandmarks_loca2_aftmas = initLandmarks_loca2 * mask
        initLandmarks_loca3 = T.alloc(initLandmarks_aftmas[67, :], 68, 2)
        initLandmarks_loca3_aftmas = initLandmarks_loca3 * mask

        weight1 = T.zeros((68, 2))
        weight1_val = T.exp(- T.sum((initLandmarks_loca1_aftmas - initLandmarks_aftmas) ** 2, axis=1) / self.sigmaV2)
        weight1 = T.set_subtensor(weight1[:, 0], weight1_val)
        weight1 = T.set_subtensor(weight1[:, 1], weight1_val)
        val1 = T.sum(weight1 * moms_aftmas, axis=0)
        dp = T.set_subtensor(dp[65, :], val1)

        weight2 = T.zeros((68, 2))
        weight2_val = T.exp(- T.sum((initLandmarks_loca2_aftmas - initLandmarks_aftmas) ** 2, axis=1) / self.sigmaV2)
        weight2 = T.set_subtensor(weight2[:, 0], weight2_val)
        weight2 = T.set_subtensor(weight2[:, 1], weight2_val)
        val2 = T.sum(weight2 * moms_aftmas, axis=0)
        dp = T.set_subtensor(dp[66, :], val2)

        weight3 = T.zeros((68, 2))
        weight3_val = T.exp(- T.sum((initLandmarks_loca3_aftmas - initLandmarks_aftmas) ** 2, axis=1) / self.sigmaV2)
        weight3 = T.set_subtensor(weight3[:, 0], weight3_val)
        weight3 = T.set_subtensor(weight3[:, 1], weight3_val)
        val3 = T.sum(weight3 * moms_aftmas, axis=0)
        dp = T.set_subtensor(dp[67, :], val3)

        deformedShape = initLandmarks_aftmas + (dp * self.tau)

        deformedShape_loca1 = T.alloc(deformedShape[65, :], 68, 2)
        deformedShape_loca2 = T.alloc(deformedShape[66, :], 68, 2)
        deformedShape_loca3 = T.alloc(deformedShape[67, :], 68, 2)

        weight11 = T.zeros((68, 2))
        weight11_val = T.exp(- T.sum((deformedShape_loca1 - deformedShape) ** 2, axis=1) / self.sigmaV2)
        weight11 = T.set_subtensor(weight11[:, 0], weight11_val)
        weight11 = T.set_subtensor(weight11[:, 1], weight11_val)
        val11 = T.sum(weight11 * moms_aftmas, axis=0)
        dp1 = T.set_subtensor(dp1[65, :], val11)

        weight22 = T.zeros((68, 2))
        weight22_val = T.exp(- T.sum((deformedShape_loca2 - deformedShape) ** 2, axis=1) / self.sigmaV2)
        weight22 = T.set_subtensor(weight22[:, 0], weight22_val)
        weight22 = T.set_subtensor(weight22[:, 1], weight22_val)
        val22 = T.sum(weight22 * moms_aftmas, axis=0)
        dp1 = T.set_subtensor(dp1[66, :], val22)

        weight33 = T.zeros((68, 2))
        weight33_val = T.exp(- T.sum((deformedShape_loca3 - deformedShape) ** 2, axis=1) / self.sigmaV2)
        weight33 = T.set_subtensor(weight33[:, 0], weight33_val)
        weight33 = T.set_subtensor(weight33[:, 1], weight33_val)
        val33 = T.sum(weight33 * moms_aftmas, axis=0)
        dp1 = T.set_subtensor(dp1[67, :], val33)

        output = (deformedShape + dp1 * self.tau).flatten()
        return output

    def get_output_for(self, input, **kwargs):

        outputs, updates = theano.scan(self.compute_landmarks_helper, input)
        return outputs
