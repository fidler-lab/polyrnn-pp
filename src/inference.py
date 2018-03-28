import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import glob
import os
import numpy as np
from PolygonModel import PolygonModel
from EvalNet import EvalNet
from GGNNPolyModel import GGNNPolygonModel
import utils
import skimage.io as io
import tqdm
import json

#
tf.logging.set_verbosity(tf.logging.INFO)
# --
flags = tf.flags
FLAGS = flags.FLAGS
# ---
flags.DEFINE_string('PolyRNN_metagraph', '', 'PolygonRNN++ MetaGraph ')
flags.DEFINE_string('PolyRNN_checkpoint', '', 'PolygonRNN++ checkpoint ')
flags.DEFINE_string('EvalNet_checkpoint', '', 'Evaluator checkpoint ')
flags.DEFINE_string('GGNN_metagraph', '', 'GGNN poly MetaGraph ')
flags.DEFINE_string('GGNN_checkpoint', '', 'GGNN poly checkpoint ')
flags.DEFINE_string('InputFolder', '../imgs/', 'Folder with input image crops')
flags.DEFINE_string('OutputFolder', '../output/', 'OutputFolder')
flags.DEFINE_boolean('Use_ggnn', False, 'Use GGNN to postprocess output')

#

_BATCH_SIZE = 1
_FIRST_TOP_K = 5

def save_to_json(crop_name, predictions_dict):
    output_dict = {'img_source': crop_name, 'polys': predictions_dict['polys'][0].tolist()}
    if 'polys_ggnn' in predictions_dict:
        output_dict['polys_ggnn'] = predictions_dict['polys_ggnn'][0].tolist()

    fname = os.path.basename(crop_name).split('.')[0] + '.json'

    fname = os.path.join(FLAGS.OutputFolder, fname)

    json.dump(output_dict, open(fname, 'w'), indent=4)


def inference(_):
    # Creating the graphs
    evalGraph = tf.Graph()
    polyGraph = tf.Graph()

    # Evaluator Network
    tf.logging.info("Building EvalNet...")
    with evalGraph.as_default():
        with tf.variable_scope("discriminator_network"):
            evaluator = EvalNet(_BATCH_SIZE)
            evaluator.build_graph()
        saver = tf.train.Saver()

        # Start session
        evalSess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True
        ), graph=evalGraph)
        saver.restore(evalSess, FLAGS.EvalNet_checkpoint)

    # PolygonRNN++
    tf.logging.info("Building PolygonRNN++ ...")
    model = PolygonModel(FLAGS.PolyRNN_metagraph, polyGraph)

    model.register_eval_fn(lambda input_: evaluator.do_test(evalSess, input_))

    polySess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True
    ), graph=polyGraph)

    model.saver.restore(polySess, FLAGS.PolyRNN_checkpoint)

    if FLAGS.Use_ggnn:
        ggnnGraph = tf.Graph()
        tf.logging.info("Building GGNN ...")
        ggnnModel = GGNNPolygonModel(FLAGS.GGNN_metagraph, ggnnGraph)
        ggnnSess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True
        ), graph=ggnnGraph)

        ggnnModel.saver.restore(ggnnSess, FLAGS.GGNN_checkpoint)

    tf.logging.info("Testing...")
    if not os.path.isdir(FLAGS.OutputFolder):
        tf.gfile.MakeDirs(FLAGS.OutputFolder)
    crops_path = glob.glob(os.path.join(FLAGS.InputFolder, '*.png'))

    for crop_path in tqdm.tqdm(crops_path):
        image_np = io.imread(crop_path)
        image_np = np.expand_dims(image_np, axis=0)
        preds = [model.do_test(polySess, image_np, top_k) for top_k in range(_FIRST_TOP_K)]

        # sort predictions based on the eval score and pick the best
        preds = sorted(preds, key=lambda x: x['scores'][0], reverse=True)[0]

        if FLAGS.Use_ggnn:
            polys = np.copy(preds['polys'][0])
            feature_indexs, poly, mask = utils.preprocess_ggnn_input(polys)
            preds_gnn = ggnnModel.do_test(ggnnSess, image_np, feature_indexs, poly, mask)
            output = {'polys': preds['polys'], 'polys_ggnn': preds_gnn['polys_ggnn']}
        else:
            output = {'polys': preds['polys']}

        # dumping to json files
        save_to_json(crop_path, output)


if __name__ == '__main__':
    tf.app.run(inference)
