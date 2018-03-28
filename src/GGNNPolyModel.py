import tensorflow as tf
import utils
import numpy as np


class GGNNPolygonModel(object):
    """Class to load GGNNPolygonModel and run inference."""

    # Tensors names to gather from the graph

    # Input
    IMG = 'imgs:0'
    FEATURE_INDEX = "feature_index:0"
    ADJ = 'adjcent:0'
    POLY = 'polys:0'
    MASK = 'masks:0'

    # Outputs
    OUTPUT_POLYS_TENSOR_NAME = 'ggnn_out_poly:0'
    OUTPUT_MASKS_TENSOR_NAME = 'ggnn_out_masks:0'

    def __init__(self, meta_graph_path, graph=None):
        """Creates and loads PolygonModel. """
        if graph is None:
            self.graph = tf.Graph()
        else:
            self.graph = graph

        self.saver = None
        self.eval_pred_fn = None
        self._restore_graph(meta_graph_path)
        self.max_poly_len = 142

    def _restore_graph(self, meta_graph_path):
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(meta_graph_path, clear_devices=True)

    def _prediction(self):
        return {
            'polys': self.graph.get_tensor_by_name(self.OUTPUT_POLYS_TENSOR_NAME),
            'masks': self.graph.get_tensor_by_name(self.OUTPUT_MASKS_TENSOR_NAME)
        }

    def register_eval_fn(self, eval_pred_fn):
        self.eval_pred_fn = eval_pred_fn

    def do_test(self, sess, input_images, feature_index, polys, masks):
        """
        Return polygon
        """

        assert input_images.shape[1:] == (224, 224, 3), 'image must be rgb 224x224 but is (%s)' % str(
            input_images.shape)

        adjcent_matrix = self.create_adjacency_matrix(polys, masks)

        pred_dict = sess.run(
            self._prediction(),
            feed_dict={
                self.IMG: input_images,
                self.FEATURE_INDEX: feature_index,
                self.ADJ: adjcent_matrix,
                self.POLY: polys,
                self.MASK: masks
            }
        )
        #
        polygons = pred_dict['polys']
        masks = pred_dict['masks']
        #
        polygons = self._postprocess_polygons(polygons, masks)
        pred_dict['polys'] = polygons

        return {'polys_ggnn': polygons}

    def create_adjacency_matrix(self, batch_poly, batch_mask):
        """
        Create adjacency matrix for ggnn

        Args:
            polygons: T x N x 2 vertices in range [0, grid_side]
            masks: T x N x 1 masks

        Returns:
            adjacency_matrix: [Batch_size, self.max_poly_len, self.max_poly_len * 3 * 2]
        """
        batch_size = len(batch_poly)
        n_nodes = self.max_poly_len
        n_edge_types = 3
        a = np.zeros([batch_size, n_nodes, n_nodes * n_edge_types * 2])
        for batch in range(len(batch_poly)):
            mask = batch_mask[batch]
            index, = np.where(mask == 0)
            if len(index) > 0:
                index = index[0]
                if index > 2:
                    for i in range(index):
                        if i % 2 == 0:
                            if i < index - 2:

                                a[batch][i][(0) * n_nodes + i + 2] = 1
                                a[batch][i + 2][(0 + n_edge_types) * n_nodes + i] = 1

                                a[batch][i + 2][(0) * n_nodes + i] = 1
                                a[batch][i][(0 + n_edge_types) * n_nodes + i + 2] = 1

                                a[batch][i][(1) * n_nodes + i + 1] = 1
                                a[batch][i + 1][(1 + n_edge_types) * n_nodes + i] = 1

                                a[batch][i + 1][(2) * n_nodes + i] = 1
                                a[batch][i][(2 + n_edge_types) * n_nodes + i + 1] = 1

                            else:
                                a[batch][i][(0) * n_nodes + 0] = 1
                                a[batch][0][(0 + n_edge_types) * n_nodes + i] = 1

                                a[batch][0][(0) * n_nodes + i] = 1
                                a[batch][i][(0 + n_edge_types) * n_nodes + 0] = 1

                                a[batch][i][(1) * n_nodes + i + 1] = 1
                                a[batch][i + 1][(1 + n_edge_types) * n_nodes + i] = 1

                                a[batch][i + 1][(2) * n_nodes + i] = 1
                                a[batch][i][(2 + n_edge_types) * n_nodes + i + 1] = 1

                        else:
                            if i < index - 1:
                                a[batch][i][(2) * n_nodes + i + 1] = 1
                                a[batch][i + 1][(2 + n_edge_types) * n_nodes + i] = 1

                                a[batch][i + 1][(1) * n_nodes + i] = 1
                                a[batch][i][(1 + n_edge_types) * n_nodes + i + 1] = 1


                            else:
                                a[batch][i][(2) * n_nodes + 0] = 1
                                a[batch][0][(2 + n_edge_types) * n_nodes + i] = 1

                                a[batch][0][(1) * n_nodes + i] = 1
                                a[batch][i][(1 + n_edge_types) * n_nodes + 0] = 1

        return a

    def _postprocess_polygons(self, polygons, masks, ):
        """
        Post process polygons.

        Args:
            polygons: T x N x 2 vertices in range [0, grid_side]
            masks: T x N x 1 masks

        Returns:
            processed_polygons: list of N polygons
        """

        result = utils._mask_polys(polygons, masks)
        result1 = [utils._poly0g_to_poly01(p, 112) for p in result]
        return result1
