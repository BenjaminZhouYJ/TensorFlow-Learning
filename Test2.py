import tensorflow as tf

sess = tf.Session()
a = tf.constant(2)
b = tf.constant(3)
print(sess.run(a+b))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))