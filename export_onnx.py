"""
Construction d'un modele onnx à partir de la version checkpoint courante d'un style 
"""

import tools.patch        # Reçoit le patch magique qui intercepte tensorflow.contrib !
import tensorflow.compat.v1 as tf
import sys
from net import generator  # Importe le générateur de votre projet AnimeGANv3

import subprocess
import sys

def convert_pb_to_onnx(command) :
    print("Conversion du modèle .pb vers .onnx")
    print(f"{command} en cours ...")
    # Prerequis : pip install tf2onnx
    try:
        # Lancement de la commande
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Conversion réussie !")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Erreur lors de la conversion :")
        print(e.stderr)


def convert_checkpoint_to_onnx(checkpoint_dir):
    tf.reset_default_graph()

    # 1. Définir l'entrée
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3], name='input_photo')

    # 2. Reconstruire le réseau (mode inférence : reuse=False, is_training=False)
    with tf.variable_scope("generator", reuse=False):
        # On récupère la sortie brute du générateur (s0)
        out_s0, _ = generator.G_net(input_photo, False)
        # Nommer le nœud de sortie pour le retrouver facilement
        output_tensor = tf.identity(out_s0, name='final_output')

    output_node_name = output_tensor.name.split(':')[0]
    print(f"Nom exact du nœud de sortie : {output_node_name}")

    # 3. Charger les poids du checkpoint
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir) # checkpoint file information
    if ckpt and ckpt.model_checkpoint_path:
        checkpoint_path = ckpt.model_checkpoint_path

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Charger votre checkpoint spécifique (ex: checkpoint/AnimeGANv3_36.ckpt)
        saver.restore(sess, checkpoint_path)
        print(f"Poids chargés depuis {checkpoint_path}")
        output_pb_path = f"{checkpoint_path}.pb"
        output_onnx_path = f"{checkpoint_path}.onnx"

        # 4. Figer le graphe (convertir les variables en constantes)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names=[output_node_name] # Utilise le nom résolu dynamiquement
        )

        # 5. Sauvegarder le fichier .pb
        with tf.gfile.GFile(output_pb_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print(f"Graphe figé sauvegardé : {output_pb_path}")

        command = [
            sys.executable,  # Utilise automatiquement le binaire Python de votre environnement actuel
            "-m",
            "tf2onnx.convert",
            "--input", output_pb_path,
            "--inputs", "input_photo:0",
            "--outputs", "generator/final_output:0",
            "--opset", "11",
            "--output", output_onnx_path
        ]
        convert_pb_to_onnx(command)

        #print(f"Creation du fichier onnx avec :")
        #print(f"python -m tf2onnx.convert --input {output_pb_path} --inputs input_photo:0 --outputs generator/final_output:0 --opset 11 --output {output_onnx_path}")


if len(sys.argv) > 1:
    model_name = sys.argv[1]

     # Modifiez ces chemins selon votre dossier
    convert_checkpoint_to_onnx(f"./checkpoint/AnimeGANv3_{model_name}")
else :
    print("Nom du model name attendu (ex Hayao, Meyer)")

"""
# Pour créer le fichier .onnx, sous (.venv) PS C:\DataScientest\Python\BD\AnimeGANv3>
python export_onnx.py Meyer

# Pour utiliser le fichier .onnx, sous (.venv) PS C:\DataScientest\Python\BD>
python.exe .\AnimeGAN_V3.py Gladiator
"""