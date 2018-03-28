import tensorflow.contrib.layers as layers
from collections import namedtuple
import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim
import poly_utils as polyutils

Inputs = namedtuple("Inputs",
                    ["cnn_feats", "pred_polys", "predicted_mask", "ious", "hidd1", "hidd2", "cells_1", "cells_2",
                     "pred_mask_imgs"])


class EvalNet(object):
    def __init__(self, batch_size, max_poly_len=71):
        self.seq_len = max_poly_len
        self.batch_size = batch_size
        self._ph = self._define_phs()
        self.cost = None
        self.predicted_ious = None
        # -
        self.name = "EvalNet"
        self.is_training = False
        self._zero_batch = np.zeros([self.batch_size, 1])
        self._first_pass = True

    def _define_phs(self):

        cnn_feats = tf.placeholder(tf.float32, shape=[self.batch_size, 28, 28, 128], name="cnn_feats")
        pred_mask_imgs = tf.placeholder(tf.float32, shape=[self.batch_size, 28, 28, 2], name="pred_mask_imgs")

        # --
        pred_polys = tf.placeholder(tf.float32, shape=[self.batch_size, self.seq_len, 2],
                                    name="pred_polys")
        predicted_mask = tf.placeholder(tf.float32, shape=[self.batch_size, self.seq_len],
                                        name="predicted_mask")
        # ---
        h1 = tf.placeholder(tf.float32, shape=[self.batch_size, self.seq_len, 28, 28, 64], name="hidden1")
        cells_1 = tf.placeholder(tf.float32, shape=[self.batch_size, 1, 28, 28, 64],
                                 name="cell_state_hidden1")
        h2 = tf.placeholder(tf.float32, shape=[self.batch_size, self.seq_len, 28, 28, 16], name="hidden2")
        cells_2 = tf.placeholder(tf.float32, shape=[self.batch_size, 1, 28, 28, 16],
                                 name="cell_state_hidden2")

        ious = tf.placeholder(tf.float32, shape=[self.batch_size, 1], name="ious")
        return Inputs(cnn_feats, pred_polys, predicted_mask, ious, h1, h2, cells_1, cells_2, pred_mask_imgs)

    def training(self):
        raise NotImplementedError()

    def draw_mask(self, img_h, img_w, pred_poly, pred_mask):
        batch_size = pred_poly.shape[0]

        pred_poly_lens = np.sum(pred_mask, axis=1)

        assert pred_poly_lens.shape[0] == batch_size == self.batch_size, '%s,%s,%s' % (
            str(pred_poly_lens.shape[0]), str(batch_size), str(self.batch_size))

        masks_imgs = []
        for i in range(batch_size):
            # Cleaning the polys
            p_poly = pred_poly[i][:pred_poly_lens[i], :]

            # Printing the mask
            # if self.draw_perimeter is False:
            try:
                mask1 = np.zeros((img_h, img_w))
                mask1 = polyutils.draw_poly(mask1, p_poly.astype(np.int))
                mask1 = np.reshape(mask1, [img_h, img_w, 1])
                # else:
                mask = polyutils.polygon_perimeter(p_poly.astype(np.int), img_side=28)
                mask = np.reshape(mask, [img_h, img_w, 1])
            except:
                import ipdb;
                ipdb.set_trace()

            mask = np.concatenate((mask, mask1), axis=2)

            masks_imgs.append(mask)
        masks_imgs = np.array(masks_imgs, dtype=np.float32)
        return np.reshape(masks_imgs, [self.batch_size, img_h, img_w, 2])

    def _feed_dict(self, train_batch, is_training=True):

        pred_polys = train_batch['raw_polys'] * np.expand_dims(train_batch['masks'], axis=2)  # (seq,batch,2)
        pred_polys = np.transpose(pred_polys, [1, 0, 2])  # (batch,seq,2)

        pred_mask = np.transpose(train_batch['masks'], [1, 0])  # (batch_size,seq_len)
        cnn_feats = train_batch['cnn_feats']  # (batch_size, 28, 28, 128)

        cells_1 = np.stack([np.split(train_batch['hiddens_list'][-1][0], 2, axis=3)[0]], axis=1)

        cells_2 = np.stack([np.split(train_batch['hiddens_list'][-1][1], 2, axis=3)[0]], axis=1)

        pred_mask_imgs = self.draw_mask(28, 28, pred_polys, pred_mask)

        if is_training:
            raise NotImplementedError()

        r = {
            self._ph.cells_1: cells_1,
            self._ph.cells_2: cells_2,
            self._ph.pred_mask_imgs: pred_mask_imgs,
            self._ph.cnn_feats: cnn_feats,
            self._ph.predicted_mask: pred_mask,
            self._ph.pred_polys: pred_polys,
            self._ph.ious: self._zero_batch
        }

        return r

    def do_train(self, sess, train_batch, cost_op, backpass_op, train_writer, log, batch_idx):
        """
        Perform a training iteration.l
        """
        raise NotImplementedError()

    def build_graph(self):
        self._build_model()
        return self.predicted_ious

    def _myForwardPass(self):
        cnn_feats = self._ph.cnn_feats
        pred_polys = self._ph.pred_polys
        pred_mask_imgs = self._ph.pred_mask_imgs
        last_cell_state_1 = self._ph.cells_1[:, -1, :, :, :]
        last_cell_state_2 = self._ph.cells_2[:, -1, :, :, :]
        weight_decay = 0.00001

        predicted_history = tf.zeros(shape=(self.batch_size, 28, 28, 1))

        # Drawing the canvas
        for i in range(self.seq_len):
            pred_polys_t = pred_polys[:, i]  # batch x
            indices = tf.concat(
                [tf.reshape(tf.range(0, self.batch_size), (self.batch_size, 1)), tf.cast(pred_polys_t, tf.int32)],
                axis=1)
            updates = tf.ones(shape=self.batch_size)
            pred_polys_t = tf.scatter_nd(indices, updates, shape=(self.batch_size, 28, 28))
            predicted_history = predicted_history + tf.expand_dims(pred_polys_t, axis=-1)

        xt = tf.concat([cnn_feats, predicted_history, pred_mask_imgs, last_cell_state_1, last_cell_state_2],
                       axis=3)

        with slim.arg_scope([slim.conv2d], kernel_size=[3, 3], stride=1,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={"is_training": self.is_training, "decay": 0.99, "center": True,
                                               "scale": True},
                            weights_initializer=layers.variance_scaling_initializer(
                                factor=2.0, mode='FAN_IN',
                                uniform=False)
                            ):
            self._conv1 = slim.conv2d(xt, scope="conv1", num_outputs=16)
            self._conv2 = slim.conv2d(self._conv1, scope="conv2", num_outputs=1)

        output = layers.fully_connected(slim.flatten(self._conv2), 1, weights_regularizer=layers.l2_regularizer(1e-5),
                                        scope="FC")
        return output

    def _build_model(self):
        prediction = self._myForwardPass()

        self.predicted_ious = prediction
        return self.predicted_ious

    def do_test(self, sess, instance, *_):
        output = sess.run(
            self.predicted_ious,
            feed_dict=self._feed_dict(instance, is_training=False)
        )

        return output
