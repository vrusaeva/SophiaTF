## General Usage [WIP]

Below is an example code snippet for training a general model with NLL loss with the Tensorflow Sophia Implementation. 

```python
import tensorflow as tf
from keras.api.losses import sparse_categorical_crossentropy
from sophia_tf import Sophia

# init model loss, input data
model = Model()
trainX, trainY = ..., ...

# init optimizer
optimizer = Sophia(lr=2e-4, betas=(0.965, 0.99), rho=0.01, weight_decay=0.2, maximize=False)

k = 10
nstep = -1

for epoch in range(epochs):
    for X, Y in zip(trainX, trainY):
        with tf.GradientTape(persistent=True) as tape:
            # Run the forward pass of the model to generate a prediction
            prediction = model(trainX, training=True)
            tape.watch(prediction)
            # Compute the training loss
            loss = loss_fcn(trainY, prediction)
            # Compute the training accuracy
            accuracy = acc_fcn(trainY, prediction)
        # Retrieve gradients of the trainable variables with respect to the training loss
        gradients = tape.gradient(loss, model.trainable_weights)
        # Update the values of the trainable variables by gradient descent
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        nstep += 1
    
        with tape:
            if nstep % k == k - 1:
                # draw y_hat from categorical distribution
                y_hat = tf.random.categorical(logits=prediction)
                # compute loss over the batch
                opt_loss = sparse_categorical_crossentropy(tf.reshape(y_hat, [-1]),
                                                           tf.reshape(prediction, [-1, prediction.shape[-1]]))
            ...
    

```
