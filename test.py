import argparse
import os,cv2
from tqdm import tqdm
from glob import glob
import time
import tools.patch        # Reçoit le patch magique qui intercepte tensorflow.contrib !
import tensorflow.compat.v1 as tf
import tf_slim as slim

import numpy as np
from net import generator
from tools.GuidedFilter import guided_filter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def save_images(images, image_path, hw):
    images = (images.squeeze()+ 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    images = cv2.resize(images, (hw[1], hw[0]))
    cv2.imwrite(image_path, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))


def preprocessing(img, x8=True):
    h, w = img.shape[:2]
    if x8: # resize image to multiple of 8s
        def to_x8s(x):
            return 256 if x < 256 else x - x%8 # if using tiny model: x - x%16
        img = cv2.resize(img, (to_x8s(w), to_x8s(h)))
    return img/127.5 - 1.0

def load_test_data(image_path, x8=True):
    img0 = cv2.imread(image_path)
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = preprocessing(img, x8)
    img = np.expand_dims(img, axis=0)
    return img, img0.shape[:2]


def sigm_out_scale(x):
    x = (x + 1.0) / 2.0
    return tf.clip_by_value(x, 0.0, 1.0)

def tanh_out_scale(x):
    x = (x - 0.5) * 2.0
    return tf.clip_by_value(x, -1.0, 1.0)

def parse_args():
    desc = "AnimeGANv3"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint/generator_v3_Hayao_weight',help='Directory name to save the checkpoints')
    parser.add_argument('--test_dir', type=str, default='inputs/imgs', help='Directory name of test photos')
    parser.add_argument('--save_dir', type=str, default='style_results/',help='Directory name of results')
    return parser.parse_args()

import onnx
from onnx import numpy_helper

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


def test(checkpoint_dir, save_dir, test_dir, onnx=""):
    # tf.reset_default_graph()
    result_dir = check_folder(save_dir)
    test_files = glob('{}/*.*'.format(test_dir))
    test_real = tf.placeholder(tf.float32, [1, None, None, 3], name='AnimeGANv3_input')
    with tf.variable_scope("generator", reuse=False):
        _, _ = generator.G_net(test_real, True)
    with tf.variable_scope("generator", reuse=True):
        test_s0, test_m = generator.G_net(test_real, False)
        test_s1 = tanh_out_scale(guided_filter(sigm_out_scale(test_s0), sigm_out_scale(test_s0), 2, 0.01))  # 0.25**2

    variables = slim.get_variables_to_restore()
    # generator_var = [var for var in variables if var.name.startswith('generator') and ('main'  in var.name  or 'base'  in var.name) and 'Adam' not in var.name and 'support' not in var.name]
    generator_var = [var for var in variables if var.name.startswith('generator') and  'Adam' not in var.name ]
    saver = tf.train.Saver(generator_var)

    # saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        # load model
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        else:
            print(" [*] Failed to find a checkpoint")
            return
        if onnx == "" :
            print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
        else :
            replace_weights_generator(sess, onnx)
            print(" [*] Success to replace weights with", onnx)     

        imgs = []
        for x in test_files:
            print("Add ", x)
            imgs.append(load_test_data(x))

        begin = time.time()
        for i, sample_file  in tqdm(list(enumerate(test_files))):
            sample_image,scale = np.asarray(imgs[i][0]),imgs[i][1]
            real, s1, s0, m = sess.run([test_real, test_s1, test_s0, test_m], feed_dict = {test_real : sample_image})
            #save_images(real, result_dir + '/a_{0}'.format(os.path.basename(sample_file)),scale)
            #save_images(s1, result_dir + '/b_{0}'.format(os.path.basename(sample_file)),scale)
            #save_images(s0, result_dir + '/c_{0}'.format(os.path.basename(sample_file)),scale)
            #save_images(m, result_dir + '/d_{0}'.format(os.path.basename(sample_file)),scale)
            save_images(m, result_dir + '/{0}'.format(os.path.basename(sample_file)), scale)
        end = time.time()
        print(f'test-time: {end-begin} s')
        print(f'one image test time : {(end-begin)/(len(test_files))} s')


if __name__ == '__main__':
    arg = parse_args()
    print(arg.checkpoint_dir)
    test(arg.checkpoint_dir, arg.save_dir, arg.test_dir, onnx="Weights/AnimeGANv3_Hayao_36.onnx")
    #test(arg.checkpoint_dir, arg.save_dir, arg.test_dir, onnx="Weights/AnimeGANv3_PortraitSketch_25.onnx")
    #test(arg.checkpoint_dir, arg.save_dir, arg.test_dir, onnx="Weights/AnimeGANv3_Shinkai_37.onnx")
    #test(arg.checkpoint_dir, arg.save_dir, arg.test_dir, onnx="checkpoint/AnimeGANv3_Meyer/model_animeganv3_meyer_64.onnx")
    #test(arg.checkpoint_dir, arg.save_dir, arg.test_dir, onnx="checkpoint/AnimeGANv3_Meyer/model_animeganv3_meyer_99.onnx")
    #test(arg.checkpoint_dir, arg.save_dir, arg.test_dir)

