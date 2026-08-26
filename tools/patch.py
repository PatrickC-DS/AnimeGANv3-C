# patch.py

import os
import logging

# Desactive les logs d'information et les avertissements de TensorFlow (0 = ALL, 1 = NO_INFO, 2 = NO_WARNINGS, 3 = NO_ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf
tf.get_logger().setLevel(logging.ERROR)  # Masque les warnings Python de TF
tf.autograph.set_verbosity(0)             # Desactive la verbosite Autograph si presente

import sys

# 1. Désactive le comportement TF2 pour rétablir Sessions, Placeholders, Graphs, etc.
tf.disable_v2_behavior()

# 2. Charge le package autonome tf-slim
import tf_slim as slim

# 3. Injecte slim sous le nom de module 'slim' pour que les "import slim" fonctionnent partout
sys.modules['slim'] = slim
