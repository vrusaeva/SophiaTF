from keras.src.optimizers import optimizer
from keras.src import ops


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
        # temp (fake, pure copy from update_hessian()) hessian update for test purposes (until actual implem)
        h.assign(beta_2 * h + (1 - beta_2) * (gradient * gradient))

        # element-wise division of momentum by Hessian value (+epsilon to prevent /0)
        # clipped to a range of +/-1 in order to prevent too-large Hessians
        # ratio = (exp_avg.abs() / (rho * bs * hess + 1e-15)).clamp(None, 1)
        # param.addcmul_(exp_avg.sign(), ratio, value=step_size_neg)
        update = ops.clip((ops.abs(m) / (rho * bs * h + 1e-15)), -1, 1)
        self.assign_add(variable, ops.sign(m) * update * (-lr))

    # Hessian estimator must be calculated within training regimen
    def update_hessian(self, gradients):
        beta_2 = self.betas[1]
        for variable in self.variables:
            grad = gradients[self._get_variable_index(variable)]
            h = self._hessian[self._get_variable_index(variable)]

            h.assign(beta_2 * h + (1 - beta_2) * (grad * grad))

    def get_config(self):
        config = super(Sophia, self).get_config()
        config.update({"learning_rate": self.learning_rate,
                       "betas": self.betas,
                       "weight_decay": self.weight_decay,
                       "rho": self.rho,
                       "batch_size": self.batch_size,
                       "maximize": self.maximize})
        return config