"""
TRANSFORME un modele .onnx en fichiers checkpoint
S'appuie sur AnimeGanv3 version Custom pour un entrainement de 0 epoque avec un chargement des poids d'un fichier onnx au début
Et un enregistrement final du checpoint 0 
"""

import subprocess
import sys

def command_execute(command) :
    print(f"{command} en cours ...")
    # Prerequis : train.py avec version AnimeGANv3_Custom
    try:
        # Lancement de la commande
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Conversion réussie !")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("Erreur lors de la conversion :")
        print(e.stderr)
        return False
    
if len(sys.argv) > 2:
    project = sys.argv[1]
    onnx_file =  sys.argv[2]

    command = [
        sys.executable,  # Utilise automatiquement le binaire Python de votre environnement actuel
        "train.py",
        "--style_dataset", project,
        "--init_G_epoch", "0",
        "--epoch", "0",
        "--load_or_resume", "load",
        "--onnx_weights_file", onnx_file
    ]
    print("Conversion du modèle onnx en checkpoint 0 du projet", project)
    if command_execute(command) :
        print(f"Checkpoint généré avec succès dans ./checkpoint/AnimeGANv3_{project}/AnimeGANv3.model-0* !")

else :
    print("Nom du projet et du fichier onnx attendus (ex Meyer Weights/AnimeGANv3_PortraitSketch_25.onnx)")
