## General Usage

Below is an example code snippet for training a general model with NLL loss with the Tensorflow Sophia Implementation. 

```python
import tensorflow as tf
import keras
from keras.api.losses import sparse_categorical_crossentropy
from sophia_tf import Sophia

# init model loss, input data, other parameters...
model = keras.Model()
trainX, trainY = ..., ...
batch_size = 64
block_length = ...

# init optimizer
optimizer = Sophia(lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=0.2, maximize=False, batch_size=batch_size * block_length)

k = 10
step = -1

for epoch in range(epochs):
    for X, Y in zip(trainX, trainY):
        # standard training code
        with tf.GradientTape(persistent=True) as tape:
            # run the forward pass of the model
            logits = model(trainX, training=True)
            tape.watch(logits)
            # compute the training loss (defined elsewhere)
            loss = loss_fcn(trainY, logits)
        # retrieve gradients of the trainable variables with respect to the training loss
        gradients = tape.gradient(loss, model.trainable_weights)
        # update the values of the trainable variables by gradient descent
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        step += 1
    
        if step % k == k - 1:
            with tape:
                # treat sequence entries as individual predictions (was testing on seq2seq task, omit if unneeded)
                logits_reshaped = tf.reshape(logits, (logits.shape[0]*logits.shape[1], logits.shape[2]))
                # draw y_hat from categorical distribution
                y_hat = tf.random.categorical(logits=logits_reshaped, num_samples=1)
                # compute loss over the batch
                # 1/B summation(ℓ(f(θ, xb), yˆb))
                opt_loss = tf.reduce_sum(sparse_categorical_crossentropy(tf.reshape(y_hat, [-1]),
                                                                         tf.reshape(logits_reshaped, [-1, logits_reshaped.shape[-1]]), 
                                                                         from_logits=True, ignore_class=-1) / (batch_size * block_length))
        # compute gradients of sampled loss over the trainable weights
        opt_grads = tape.gradient(opt_loss, model.trainable_weights)
        optimizer.update_hessian(opt_grads, model.trainable_weights)
```
