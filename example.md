NOTE: I am not certain that this works as intended on large models. Any testing and feedback is appreciated!

## General Usage

To install, simply download sophia_tf.py from this repo.

Below is an example code snippet for training a general model with NLL loss with the Tensorflow SophiaG implementation.
Note that update_hessian_auto_g() is not yet tested - use the logic below it and adapt for your specific task if needed.

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
track_clipping = True

# init optimizer
optimizer = Sophia(lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=0.2, maximize=False, track_clipping=track_clipping, batch_size=batch_size * block_length)

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
    
        # Hessian update logic; ensure logits are of shape (batch_size * block_length, num_classes)
        if step % k == k - 1:
            optimizer.update_hessian_auto_g(tape, logits, model.trainable_weights) 
                                          
            # If the above does not work, use the direct implementation instead: 
            with tape:
                # draw y_hat from categorical distribution
                y_hat = tf.random.categorical(logits=logits, num_samples=1)
                # compute loss over the batch
                # 1/B summation(ℓ(f(θ, xb), yˆb))
                opt_loss = tf.reduce_sum(sparse_categorical_crossentropy(tf.reshape(y_hat, [-1]),
                                                                         tf.reshape(logits, [-1, logits.shape[-1]]), 
                                                                         from_logits=True, ignore_class=-1) / (batch_size * block_length))
            # compute gradients of sampled loss over the trainable weights
            opt_grads = tape.gradient(opt_loss, model.trainable_weights)
            optimizer.update_hessian(opt_grads, model.trainable_weights)
    # Tracking of clipped steps for parameter tuning (optional, see README.md)
    # Should be ~equal to 1 - train/win_rate
    if track_clipping:
        print(f"Clip rate {optimizer.clip_count/optimizer.step}")
```

Replace Hessian update logic with the following if using SophiaH.
```python
    # Hessian update logic
        if step % k == k - 1:
            optimizer.update_hessian_auto_h(tape, gradients, model.trainable_weights) 
                                          
            # If the above does not work, use the direct implementation instead: 
            with tape:
                # take a vector u of same shape as gradients from spherical normal distribution
                u = tf.random.normal(gradients.shape, dtype=gradients.dtype)
                # element-wise product of Hessian with vector u = (∇^2)ℓ(θ)u
                dp = tf.reduce_sum(gradients * u)
            # ∇(⟨∇L(θ), u⟩)
            hessian_vector_product = tape.gradient(dp, model.trainable_weights)
            optimizer.update_hessian(u * hessian_vector_product, model.trainable_weights)
```
## Planned Updates:
- Ensure internally-implemented Hessian updates (Hutchinson and GNB) work as expected.
- Generalize internal Hessian update (GNB) so loss functions other than categorical crossentropy can be used.
- Set up installation of this optimizer via pip.

If you have any, please submit any feedback or additional features you would like to see.
