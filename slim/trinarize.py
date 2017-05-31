# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains methods to trinarize convolutional layers"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope

def where_func( cond, t, e ):
    if tf.__version__ == '0.11.0rc0':
        return tf.select( cond, t, e )
    return tf.where( cond, t, e )

def mul_func( a, b ):
    if tf.__version__ == '0.11.0rc0':
        return tf.mul( a, b )
    return tf.multiply( a, b )

def trinarize( x, use_sparsity = False, eta = 0.9 ):
    clip_val = tf.clip_by_value( x, -1, 1 )
    x_shape = x.get_shape()
    if use_sparsity:
        E = tf.reduce_mean(tf.abs(x))
    else:
        E = tf.constant( 1.0 )
    one = tf.constant( 1.0, shape = x_shape )
    t_x = where_func( tf.less_equal( clip_val, -eta ), mul_func( one, -E ), clip_val )
    t_x = where_func( tf.greater_equal( clip_val, eta ), mul_func( one, E ), t_x )
    tri_out = where_func( tf.logical_and( tf.greater( clip_val, -eta ), tf.less( clip_val, eta ) ),
                          tf.constant( 0.0, shape = x_shape ), t_x )
    # use the stop gradient trick to have identity backprop to x
    return x + tf.stop_gradient( tri_out - x )

def ttq_method( x, thre, w_p, w_n ):
    thre_x = tf.stop_gradient(tf.reduce_max(tf.abs(x)) * thre)
    x_shape = x.get_shape()

    one = tf.constant( 1.0, shape = x_shape )
    t_x = where_func( tf.less_equal( x, -thre_x ), mul_func( one, w_n ), x )
    t_x = where_func( tf.greater_equal( x, thre_x ), mul_func( one, w_p ), t_x )
    mask_z = where_func( tf.logical_and( tf.greater( x, -thre_x ), tf.less( x, thre_x ) ),
                      tf.constant( 0.0, shape = x_shape ), one )

    with tf.get_default_graph().gradient_override_map({"Sign": "Identity", "Mul": "Add"}):
        w =  tf.sign(x) * tf.stop_gradient(mask_z)

    return w * t_x

def replace_get_variable( use_sparsity = False, use_multiplicative = False, tri_eta = 0.9, use_ttq = 0.0 ):
    old_getv = tf.get_variable
    old_vars_getv = variable_scope.get_variable

    def get_variable_override(name, shape=None, **kwargs):
        v = old_getv(name, shape, **kwargs)
        # only trinarize the conv weights not biases
        if "weights" in v.name and "Logits" not in v.name:
            tf.logging.info( "trinarizing: " + v.name )
            if use_ttq != 0.0:
                # declare the pos and neg vars
                w_p = old_getv('Wp', collections=[tf.GraphKeys.VARIABLES, 'positives'], initializer=1.0)
                w_n = old_getv('Wn', collections=[tf.GraphKeys.VARIABLES, 'negatives'], initializer=1.0)
                tri_out = ttq_method( v, use_ttq, w_p, w_n )
            else:
                tri_out = trinarize(v, use_sparsity = use_sparsity, eta = tri_eta )
                if use_multiplicative:
                    # allow gradient on mul through
                    m = old_getv( name + "_mul", [1], **kwargs, initializer=1.0 )
                    tri_out = mul_func( tri_out, m )
            tf.add_to_collection( "trinarized_out", tri_out )
            return tri_out
        return v

    def undo():
        tf.get_variable = old_getv
        variable_scope.get_variable = old_vars_getv
    
    tf.get_variable = get_variable_override
    variable_scope.get_variable = get_variable_override

    return undo
