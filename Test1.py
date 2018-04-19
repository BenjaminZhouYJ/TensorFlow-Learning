import tensorflow as tf
hello = tf.constant("I love you Julie!")
sess = tf.Session()
print(sess.run(hello))
