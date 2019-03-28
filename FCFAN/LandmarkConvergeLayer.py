from lasagne.layers import MergeLayer
import theano
class LandmarkConvergeLayer(MergeLayer):
    def __init__(self, landmarks_Org1, landmarks_Org2, landmarks_Org3, landmarks_Org4, landmarks_Org5, landmarks_Org6,
                 landmarks_Org7, landmarks_Org8, landmarks_Org11, landmarks_Org12,
                 landmarks_Org13, landmarks_Org14, **kwargs):

        super(LandmarkConvergeLayer, self).__init__([landmarks_Org1, landmarks_Org2, landmarks_Org3, landmarks_Org4,
                                                     landmarks_Org5, landmarks_Org6, landmarks_Org7, landmarks_Org8,
                                                     landmarks_Org11, landmarks_Org12,
                                                     landmarks_Org13, landmarks_Org14], **kwargs)

    def get_output_shape_for(self, input_shapes):
        # output_shape = list(input_shapes[0])
        # return tuple(output_shape)
        return (None, 136)

    def converge_helper(self, landmarksOrg1, landmarksOrg2, landmarksOrg3, landmarksOrg4,
                                                     landmarksOrg5, landmarksOrg6, landmarksOrg7, landmarksOrg8,
                                                     landmarksOrg11, landmarksOrg12,
                                                     landmarksOrg13, landmarksOrg14):

        output = landmarksOrg1 + landmarksOrg2 + landmarksOrg3 + landmarksOrg4 + landmarksOrg5 + landmarksOrg6 + \
                 landmarksOrg7 + landmarksOrg8 + landmarksOrg11 + landmarksOrg12 + \
                 landmarksOrg13 + landmarksOrg14

        return output

    def get_output_for(self, inputs, **kwargs):
        outputs, updates = theano.scan(self.converge_helper, inputs)

        return outputs
