# TRANSFORME un modele .onnx en fichiers checkpoint

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import onnx
from onnx import numpy_helper
import sys
from net import generator

def print_vars(onnx_weights, tf) :
    # Affiche la liste des variables créées par TensorFlow
    tf_vars = [v.name.split(':')[0] for v in tf.trainable_variables()]
    print(f"\n--- Total variables TF : {len(tf_vars)} ---")
    print("Exemples de noms TF :", tf_vars)

    # Affiche la liste des tenseurs d'initialisation dans l'ONNX
    onnx_keys = list(onnx_weights.keys())
    print(f"\n--- Total poids ONNX : {len(onnx_keys)} ---")
    print("Exemples de noms ONNX :", onnx_keys)

def load_weights_onnx(filename_onnx) :
        # -------------------------------------------------------------------------
        # Charger les poids depuis le fichier ONNX
        # -------------------------------------------------------------------------
        onnx_model = onnx.load(filename_onnx)
        onnx_weights = {}

        for initializer in onnx_model.graph.initializer:
            weight = numpy_helper.to_array(initializer)
            # Transposition NCHW (ONNX) -> NHWC (TensorFlow) pour les convolutions (4D)
            if weight.ndim == 4:
                weight = np.transpose(weight, (2, 3, 1, 0))
            onnx_weights[initializer.name] = weight

        return onnx_weights

def replace_weights_generator(sess, filename_onnx) :
    # -------------------------------------------------------------------------
    # ÉCRASER uniquement les poids du Générateur avec ceux de l'ONNX
    # -------------------------------------------------------------------------
    onnx_weights = load_weights_onnx(filename_onnx)

    # Récupérer uniquement les variables du Générateur
    gen_vars = [
        v
        for v in tf.compat.v1.trainable_variables()
        if "generator" in v.name.lower()
    ]

    replaced_count = 0
    for var in gen_vars:
        var_name_clean = var.name.split(":")[0]

        # Chercher la correspondance dans les poids ONNX
        for onnx_name, weight_data in onnx_weights.items():
            if onnx_name in var_name_clean or var_name_clean in onnx_name:
                try:
                    # Écriture directe dans la variable en session
                    sess.run(var.assign(weight_data))
                    replaced_count += 1
                    break
                except Exception as e:
                    print(f"Erreur pour {var.name} : {e}")

    print(
        f"Succès : {replaced_count} variables du Générateur ont été remplacées par l'ONNX !"
    )        

def convert_onnx_to_checkpoint(model_name, version) :
    # Construire le graphe TensorFlow
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3], name='input_photo')
    with tf.variable_scope("generator", reuse=False):
        # On récupère la sortie brute du générateur (s0)
        out_s0, _ = generator.G_net(input_photo, False)
        # Nommer le nœud de sortie pour le retrouver facilement
        output_tensor = tf.identity(out_s0, name='final_output')

    # Charger et assigner les poids ONNX aux variables TensorFlow
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        replace_weights_generator(sess, f"Weights/AnimeGANv3_{model_name}_{version}.onnx")
                
        # Sauvegarder au format Checkpoint
        saver.save(sess, f"checkpoint/AnimeGANv3_{model_name}/AnimeGANv3.model-{version}")
        print(f"Checkpoint généré avec succès dans ./checkpoint/AnimeGANv3_{model_name}/AnimeGANv3.model-{version}* !")


def old_convert_onnx_to_checkpoint(model_name, version) :
    # 2. Charger le modèle ONNX
    onnx_model = onnx.load(f"Weights/AnimeGANv3_{model_name}_{version}.onnx")
    onnx_weights = {initializer.name: numpy_helper.to_array(initializer) for initializer in onnx_model.graph.initializer}

    # 3. Construire le graphe TensorFlow
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3], name='input_photo')
    with tf.variable_scope("generator", reuse=False):
        # On récupère la sortie brute du générateur (s0)
        out_s0, _ = generator.G_net(input_photo, False)
        # Nommer le nœud de sortie pour le retrouver facilement
        output_tensor = tf.identity(out_s0, name='final_output')

    # 4. Charger et assigner les poids ONNX aux variables TensorFlow
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print_vars(onnx_weights, tf)
        # Parcourir les variables TF et leur associer les poids ONNX correspondants
        for var in tf.trainable_variables():
            print(var.name)
            var_name_clean = var.name.split(':')[0]    

            # Chercher la correspondance dans les poids ONNX
            for onnx_name, weight_data in onnx_weights.items():
                if onnx_name in var_name_clean or var_name_clean in onnx_name:
                    if len(weight_data.shape) == 4:
                        weight_data = np.transpose(weight_data, (2, 3, 1, 0)) # CORRECTION DE L'AXE (Évite les images vertes)
                    # Écriture directe dans la variable en session
                    sess.run(var.assign(weight_data))
                    print(f"Chargé : {var_name_clean}")
                
        # 5. Sauvegarder au format Checkpoint
        saver.save(sess, f"checkpoint/AnimeGANv3_{model_name}/AnimeGANv3.model-{version}")
        print(f"Checkpoint généré avec succès dans ./checkpoint/AnimeGANv3_{model_name}/AnimeGANv3.model-{version}* !")

    
if len(sys.argv) > 2:
    model_name = sys.argv[1]
    version =  sys.argv[2]

    convert_onnx_to_checkpoint(model_name, version)
else :
    print("Nom du model name et version attendus (ex PortraitSketch 25)")
