import tensorflow as tf
import tensorbayes as tb
import numpy as np
from codebase.args import args
from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_xent_two
from tensorflow.contrib.framework import add_arg_scope

@add_arg_scope
def normalize_perturbation(d, scope=None):
    with tf.name_scope(scope, 'norm_pert'):
        output = tf.nn.l2_normalize(d, axis=list(range(1, len(d.shape))))
    return output

@add_arg_scope
def scale_gradient(x, scale, scope=None, reuse=None):
    with tf.name_scope('scale_grad'):
        output = (1 - scale) * tf.stop_gradient(x) + scale * x
    return output

@add_arg_scope
def noise(x, std, phase, scope=None, reuse=None):
    with tf.name_scope(scope, 'noise'):
        eps = tf.random_normal(tf.shape(x), 0.0, std)
        output = tf.where(phase, x + eps, x)
    return output

@add_arg_scope
def leaky_relu(x, a=0.2, name=None):
    with tf.name_scope(name, 'leaky_relu'):
        return tf.maximum(x, a * x)

@add_arg_scope
def basic_accuracy(a, b, scope=None):
    with tf.name_scope(scope, 'basic_acc'):
        a = tf.argmax(a, 1)
        b = tf.argmax(b, 1)
        eq = tf.cast(tf.equal(a, b), 'float32')
        output = tf.reduce_mean(eq)
    return output

@add_arg_scope
def perturb_image(x, p, classifier, pert='vat', scope=None, prob=None):
    with tf.name_scope(scope, 'perturb_image'):
        eps = 1e-6 * normalize_perturbation(tf.random_normal(shape=tf.shape(x)))

        # Predict on randomly perturbed image
        eps_p, _ = classifier(x + eps, trim=0, phase=True, reuse=True, prob=prob)
        loss = softmax_xent_two(labels=p, logits=eps_p) # again uses this function since 
                                                        # "p" is converted to label 

        # Based on perturbed image, get direction of greatest error
        # derivada de loss con respecto a eps
        # agregation method EXPERIMENTAL_ACCUMULATE_N
        eps_adv = tf.gradients(loss, [eps], aggregation_method=2)[0]

        # Use that direction as adversarial perturbation
        eps_adv = normalize_perturbation(eps_adv)
        # When building ops to compute gradients, this op prevents the contribution 
        # of its inputs to be taken into account. 
        x_adv = tf.stop_gradient(x + args.radius * eps_adv)

    return x_adv

@add_arg_scope
def vat_loss(x, p, classifier, scope=None, prob=None):
    with tf.name_scope(scope, 'smoothing_loss'):
        # apply perturb
        # receive sample x, prediction, classifier
        x_adv = perturb_image(x, p, classifier, prob=prob)
        # obtains Virtual Loss
        p_adv, _ = classifier(x_adv, trim=0, phase=True, reuse=True, prob=prob)
        loss = tf.reduce_mean(softmax_xent_two(labels=tf.stop_gradient(p), logits=p_adv))

    return loss

