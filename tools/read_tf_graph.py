import tensorflow as tf
import sys
import argparse
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

parser = argparse.ArgumentParser(description='Display TensorFlow graph into TensorBoard')
parser.add_argument('tf_model', metavar='tf_model', type=str, help='tf model path to be loaded')
parser.add_argument('logs_dir', metavar='logs_dir', type=str, help='tensorboard logs directory')
args = parser.parse_args()

with tf.Session() as sess:
    model_filename = args.tf_model
    with gfile.FastGFile(model_filename, 'rb') as f:

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)

    LOGDIR = args.logs_dir
    train_writer = tf.summary.FileWriter(LOGDIR)
    train_writer.add_graph(sess.graph)
    print("Model Imported. Visualize by running > tensorboard --logdir={}".format(LOGDIR))
