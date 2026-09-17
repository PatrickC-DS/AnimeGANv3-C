# patch.py
import sys
import tensorflow.compat.v1 as tf
import tensorflow.contrib as tf_contrib  # Fonctionnera grâce à patch.py

# 1. Désactive le comportement TF2 pour rétablir Sessions, Placeholders, Graphs, etc.
tf.disable_v2_behavior()

# 2. Charge le package autonome tf-slim
import tf_slim as slim

# 3. Injecte slim sous le nom de module 'slim' pour que les "import slim" fonctionnent partout
sys.modules['slim'] = slim





import os
# 1. Coupe les logs C++ de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 2. Désactive les logs de la bibliothèque ABSL (utilisée par TF)
os.environ['ABSL_LOGGING_MIN_INFO_LEVEL'] = '3'

import logging
# 3. Coupe tous les loggers Python associés à TF ou Keras
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import warnings
# 4. Coupe les warnings standards
warnings.filterwarnings("ignore")

# 5. Désactive spécifiquement les avertissements de retracing de TF
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)