import os, sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np

class ImageGenerator(object):

    def __init__(self, image_dir, image_size, batch_size, num_cpus=8, is_grayscale=False):
        self.paths = self.get_image_paths_train(image_dir)
        self.num_images = len(self.paths)
        self.num_cpus = num_cpus
        self.size = image_size
        self.batch_size = batch_size
        self.is_grayscale = is_grayscale  # Flag pour basculer N&B / Couleur

    def get_image_paths_train(self, image_dir):
        paths = []
        i = 0
        for path in os.listdir(image_dir):
            if path.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png']:
                continue
            path_full = os.path.join(image_dir, path)
            if not os.path.isfile(path_full):
                continue
            paths.append(path_full)
            i += 1
            if i > 1000 :   # Fix pour rÃ©duire le dataset et donc le temps d'entrainement
                return paths
        return paths

    def read_image(self, img_path):
        path_str = img_path.decode()

        if 'style' in path_str or 'smooth' in path_str:
            # Lecture multi-format (1 ou 3 canaux)
            image = cv2.imread(path_str, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Impossible de lire : {path_str}")
                
            if len(image.shape) == 2:
                # Si l'image est physiquement à 1 canal (N&B) -> Conversion RGB (R=G=B)
                image1 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
            else:
                # Si l'image a 3 canaux (Couleur ou N&B 3 canaux)
                image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
                
                # N&B strict UNIQUEMENT si spécifié
                if self.is_grayscale:
                    gray = np.mean(image1, axis=-1, keepdims=True)
                    image1 = np.repeat(gray, 3, axis=-1)

            image2 = np.zeros(image1.shape, dtype=np.float32)

        else:
            # Real photos (toujours en couleur RGB)
            image = cv2.imread(path_str)
            image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            
            seg_path = path_str.replace('train_photo', "seg_train_5-0.8-50")
            image_seg = cv2.imread(seg_path)
            
            if image_seg is not None:
                image2 = cv2.cvtColor(image_seg, cv2.COLOR_BGR2RGB).astype(np.float32)
            else:
                image2 = np.zeros_like(image1, dtype=np.float32)

        return image1, image2

    def process_image(self, img_path):
        image1, image2 = self.read_image(img_path)
        processing_image1 = image1 / 127.5 - 1.0
        processing_image2 = image2 / 127.5 - 1.0
        return processing_image1, processing_image2

    def load_images(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.paths)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=len(self.paths))
        dataset = dataset.map(lambda img_path: tf.py_func(self.process_image, [img_path], [tf.float32, tf.float32]), self.num_cpus)
        dataset = dataset.batch(self.batch_size)
        return dataset.make_one_shot_iterator().get_next()