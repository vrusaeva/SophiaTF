from keras.src.optimizers import optimizer
from keras.src import ops
from keras.src.losses import sparse_categorical_crossentropy
import tensorflow as tf


# Keras 3 implementation of Sophia optimizer (WIP.)
# Based on https://github.com/Liuhong99/Sophia.
class Sophia(optimizer.Optimizer):
    def __init__(self, lr=1e-4, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1, *, maximize: bool = False,
                 batch_size=64, name="Sophia", **kwargs):
        super().__init__(learning_rate=lr, name=name, **kwargs)
        self.betas = betas
        self.rho = rho
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.maximize = maximize  # if wishing to maximize loss
        # CUDA is not currently supported/used
        # self.capturable = capturable

    # Sophia has state variables for the EMA of the first moment and EMA of the Hessian matrix.
    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self._fmoment = []
        self._hessian = []
        for i, var in enumerate(var_list):
            self._fmoment.append(
                self.add_variable(shape=var.shape, initializer="zeros", name=f"fmoment_{i}")
            )
            self._hessian.append(
                self.add_variable(shape=var.shape, initializer="zeros", name=f"hessian_{i}")
            )
        self.step = self.add_variable(shape=(), initializer="zeros", name="step")

    def update_step(self, gradient, variable, learning_rate):
        # set variables for easier in-function use
        lr = ops.cast(learning_rate, variable.dtype)
        gradient = ops.cast(gradient, variable.dtype)
        beta_1 = self.betas[0]
        beta_2 = self.betas[1]
        rho = self.rho
        weight_decay = self.weight_decay
        maximize = self.maximize
        bs = self.batch_size

        # get optimizer state variables
        m = self._fmoment[self._get_variable_index(variable)]
        h = self._hessian[self._get_variable_index(variable)]
        step = self.step

        # increment step counter
        step.assign(step + 1)

        # if maximizing loss, negate gradients
        if maximize:
            gradient = -gradient

        # perform stepweight decay
        variable.assign(variable * (1 - weight_decay * lr))

        # update first moment moving average
        m.assign(beta_1 * m + (1 - beta_1) * gradient)

        # element-wise division of momentum by Hessian value (+epsilon to prevent /0)
        # clipped to a range of +/-1 in order to prevent too-large updates
        # ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
        # param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
        update = ops.clip((ops.abs(m) / (tf.math.maximum(rho * bs * h, 1e-15))), -1, 1)
        self.assign_add(variable, ops.sign(m) * update * (-lr))

    # Hessian estimator must be calculated within training regimen
    def update_hessian(self, gradients, variables):
        beta_2 = self.betas[1]
        for variable in variables:
            grad = gradients[self._get_variable_index(variable)]
            h = self._hessian[self._get_variable_index(variable)]

            h.assign(beta_2 * h + (1 - beta_2) * (grad * grad))

    # Hutchinson estimator - independent of loss function
    # Ensure your GradientTape is persistent to use
    def update_hessian_auto_h(self, tape, gradients, variables):
        with tape:
            # take a vector u of same shape as gradients from spherical normal distribution
            u = tf.random.normal(gradients.shape, dtype=gradients.dtype)
            # element-wise product of Hessian with vector u = (∇^2)ℓ(θ)u
            dp = tf.reduce_sum(gradients * u)
        # ∇(⟨∇L(θ), u⟩)
        hessian_vector_product = tape.gradient(dp, variables)
        self.update_hessian(u * hessian_vector_product, variables)

    # Gauss-Newton-Bartlett estimator
    # Ensure your GradientTape is persistent and logits are of the correct shape (batch_size, num_classes) to use
    def update_hessian_auto_g(self, tape, logits, variables):
        with tape:
            # draw y_hat from categorical distribution
            y_hat = tf.random.categorical(logits=logits, num_samples=1)
            # compute loss over the batch
            # 1/B summation(ℓ(f(θ, xb), yˆb))
            opt_loss = tf.reduce_sum(sparse_categorical_crossentropy(tf.reshape(y_hat, [-1]),
                                                                     tf.reshape(logits, [-1, logits.shape[-1]]),
                                                                     from_logits=True, ignore_class=-1) / self.batch_size)
        # compute gradients of sampled loss over the trainable weights
        opt_grads = tape.gradient(opt_loss, variables)
        self.update_hessian(opt_grads, variables)

    def get_config(self):
        config = super(Sophia, self).get_config()
        config.update({"learning_rate": self.learning_rate,
                       "betas": self.betas,
                       "weight_decay": self.weight_decay,
                       "rho": self.rho,
                       "batch_size": self.batch_size,
                       "maximize": self.maximize})
        return config