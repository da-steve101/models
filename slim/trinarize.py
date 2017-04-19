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

def trinarize( x ):
    eta = 0.9
    with tf.get_default_graph().gradient_override_map({
            "clip_by_value" : "Identity",
            "where": "Identity",
            "select": "Identity",
            "less" : "Identity",
            "greater": "Identity",
            "less_equal" : "Identity",
            "logical_and" : "Identity",
            "greater_equal": "Identity"}):
        clip_val = tf.clip_by_value( x, -1, 1 )
        x_shape = x.get_shape()
        E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
        t_x = where_func( tf.less_equal( clip_val, -eta ), -E, clip_val )
        t_x = where_func( tf.greater_equal( clip_val, eta ), E, t_x )
        return where_func( tf.logical_and( tf.greater( clip_val, -eta ), tf.less( clip_val, eta ) ),
                           tf.constant( 0.0, shape = x_shape ), t_x )

def replace_get_variable():
    old_getv = tf.get_variable
    old_vars_getv = variable_scope.get_variable

    def get_variable_override(name, shape=None, **kwargs):
        v = old_getv(name, shape, **kwargs)
        # only trinarize the conv weights not biases
        if "weights" in v.name:
            tf.logging.info( "trinarizing: " + v.name )
            return trinarize(v)
        return v

    def undo():
        tf.get_variable = old_getv
        variable_scope.get_variable = old_vars_getv
    
    tf.get_variable = get_variable_override
    variable_scope.get_variable = get_variable_override

    return undo
