import tensorflow as tf
import numpy as np
import utils
from distutils.version import LooseVersion

class PolygonModel(object):
    """Class to load PolygonModel and run inference."""

    # Tensors names to gather from the graph

    # Input
    INPUT_IMGS_TENSOR_NAME = 'InputImgs:0'
    INPUT_FIRST_TOP_K = "TopKFirstPoint:0"

    # Outputs
    OUTPUT_POLYS_TENSOR_NAME = 'OutputPolys:0'
    OUTPUT_MASKS_TENSOR_NAME = 'OutputMasks:0'
    OUTPUT_CNN_FEATS_TENSOR_NAME = 'OutputCNNFeats:0'
    # --
    OUTPUT_STATE1_TENSOR_NAME = 'OutputState1:0'
    OUTPUT_STATE2_TENSOR_NAME = 'OutputState2:0'

    def __init__(self, meta_graph_path, graph=None):
        """Creates and loads PolygonModel. """

        #check whether a supported version of tensorflow is installed
        if (
                (LooseVersion(tf.__version__) < LooseVersion('1.3.0')) 
             or (LooseVersion(tf.__version__) > LooseVersion('1.3.1'))
           ):
            err_string = 'you are using tensorflow version ' + tf.__version__ + ' but only versions 1.3.0 to 1.3.1 are supported'
            raise NotImplementedError(err_string)

        if graph is None:
            self.graph = tf.Graph()
        else:
            self.graph = graph

        self.saver = None
        self.eval_pred_fn = None
        self._restore_graph(meta_graph_path)

    def _restore_graph(self, meta_graph_path):
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(meta_graph_path, clear_devices=True)

    def _prediction(self):
        return {
            'polys': self.graph.get_tensor_by_name(self.OUTPUT_POLYS_TENSOR_NAME),
            'masks': self.graph.get_tensor_by_name(self.OUTPUT_MASKS_TENSOR_NAME),
            'state1': self.graph.get_tensor_by_name(self.OUTPUT_STATE1_TENSOR_NAME),
            'state2': self.graph.get_tensor_by_name(self.OUTPUT_STATE2_TENSOR_NAME),
            'cnn_feats': self.graph.get_tensor_by_name(self.OUTPUT_CNN_FEATS_TENSOR_NAME)
        }

    def register_eval_fn(self, eval_pred_fn):
        self.eval_pred_fn = eval_pred_fn

    def do_test(self, sess, input_images, first_top_k=0):
        """
        Return polygon
        """
        assert input_images.shape[1:] == (224, 224, 3), 'image must be rgb 224x224 (%s)' % str(input_images.shape)
        pred_dict = sess.run(
            self._prediction(),
            feed_dict={self.INPUT_IMGS_TENSOR_NAME: input_images, self.INPUT_FIRST_TOP_K: first_top_k}
        )
        #
        polygons = pred_dict['polys']
        pred_dict['raw_polys'] = polygons
        masks = pred_dict['masks']
        #
        polygons = self._postprocess_polygons(polygons, masks)
        pred_dict['polys'] = polygons
        pred_dict['hiddens_list'] = [[pred_dict['state1'], pred_dict['state2']]]

        if self.eval_pred_fn is not None:
            scores = self.eval_pred_fn(pred_dict)
        else:
            scores = None

        return {'polys': polygons, 'scores': scores}

    def _postprocess_polygons(self, polygons, masks, ):
        """
        Post process polygons.

        Args:
            polygons: T x N x 2 vertices in range [0, grid_side]
            masks: T x N x 1 masks

        Returns:
            processed_polygons: list of N polygons
        """
        result = np.swapaxes(polygons, 0, 1)
        masks = np.swapaxes(masks, 0, 1)
        result = utils._mask_polys(result, masks)
        result = [utils._poly0g_to_poly01(p) for p in result]

        return result
