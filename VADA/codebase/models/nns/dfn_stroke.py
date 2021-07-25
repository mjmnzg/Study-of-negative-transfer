import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorbayes.layers import dense, batch_norm
from codebase.args import args
from codebase.normalization import instance_norm
from tensorflow.python.layers.core import dropout

def classifier(x, phase, enc_phase=1, trim=0, scope='class', 
                reuse=None, internal_update=False, 
                getter=None, prob=None):
                    
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        
        
        with arg_scope([dense], bn=True, phase=phase), \
            arg_scope([batch_norm], internal_update=internal_update):
            
            x = tf.layers.flatten(x)
            #preprocess = instance_norm if args.inorm else tf.identity
                
            layout = [
                #(preprocess, (), {}),
                (dense, (100,), dict(activation=tf.nn.relu)),
                (dropout, (), dict(training=phase)),
                (dense, (100,), dict(activation=tf.nn.relu)),
                (dropout, (), dict(training=phase)),
                (dense, (100,), dict(activation=tf.nn.relu)),
                (dropout, (), dict(training=phase)),
                (dense, (args.Y,), dict(activation=None))
            ]

            # FEATURE EXTRACTION LAYER
            # NOTE: it depends on variable trim, because trim=0 obtains all network
            # trim = 0 obtains "output layer"
            # trim = 1 obtains "dropout layer"
            # trim = 2 obtains "dense3 layer"
            # trim = 3 obtains "dropout layer"
            # trim = 4 obtains "dense2 layer"
            
            if enc_phase:
                start = 0
                end = len(layout) - trim
            else: # Obtain classifier layers
                start = len(layout) - trim
                end = len(layout)
            
            layers = {}
            for i in range(start, end):
                with tf.variable_scope('l{:d}'.format(i)):
                    f, f_args, f_kwargs = layout[i]
                    x = f(x, *f_args, **f_kwargs)
                    layers["feature"] = x
            
            # assign output layer to dictionary
            layers["output"] = layout[-1]
            
        
    return x, layers

def feature_discriminator(x, phase, C=1, reuse=None):
    with tf.variable_scope('disc/feat', reuse=reuse):
        with arg_scope([dense], activation=tf.nn.relu):
            x = dense(x, 100)
            x = dense(x, C, activation=None)

    return x
