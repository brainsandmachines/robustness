import torch as ch
import torch.nn as nn
import time
import os

if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

from .tools import helpers
from . import attack_steps

STEPS = {
    'inf': attack_steps.LinfStep,
    '2': attack_steps.L2Step,
    'unconstrained': attack_steps.UnconstrainedStep,
    'fourier': attack_steps.FourierStep,
    'random_smooth': attack_steps.RandomStep
}

class Attacker(ch.nn.Module):
    """
    Attacker class, used to make adversarial examples.

    This is primarily an internal class, you probably want to be looking at
    :class:`robustness.attacker.AttackerModel`, which is how models are actually
    served (AttackerModel uses this Attacker class).

    However, the :meth:`robustness.Attacker.forward` function below
    documents the arguments supported for adversarial attacks specifically.
    """
    def __init__(self, model, dataset):
        """
        Initialize the Attacker

        Args:
            nn.Module model : the PyTorch model to attack
            Dataset dataset : dataset the model is trained on, only used to get mean and std for normalization
        """
        super(Attacker, self).__init__()
        self.normalize = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model

    def forward(self, x, target, *_, constraint, eps, step_size, iterations,
                random_start=False, random_restarts=False, do_tqdm=False,
                targeted=False, custom_loss=None, should_normalize=True,
                orig_input=None, use_best=True, return_image=True,
                est_grad=None, mixed_precision=False):
        """
        Implementation of forward (finds adversarial examples). Note that
        this does **not** perform inference and should not be called
        directly; refer to :meth:`robustness.attacker.AttackerModel.forward`
        for the function you should actually be calling.

        Args:
            x, target (ch.tensor) : see :meth:`robustness.attacker.AttackerModel.forward`
            constraint
                ("2"|"inf"|"unconstrained"|"fourier"|:class:`~robustness.attack_steps.AttackerStep`)
                : threat model for adversarial attacks (:math:`\\ell_2` ball,
                :math:`\\ell_\\infty` ball, :math:`[0, 1]^n`, Fourier basis, or
                custom AttackerStep subclass).
            eps (float) : radius for threat model.
            step_size (float) : step size for adversarial attacks.
            iterations (int): number of steps for adversarial attacks.
            random_start (bool) : if True, start the attack with a random step.
            random_restarts (bool) : if True, do many random restarts and
                take the worst attack (in terms of loss) per input.
            do_tqdm (bool) : if True, show a tqdm progress bar for the attack.
            targeted (bool) : if True (False), minimize (maximize) the loss.
            custom_loss (function|None) : if provided, used instead of the
                criterion as the loss to maximize/minimize during
                adversarial attack. The function should take in
                :samp:`model, x, target` and return a tuple of the form
                :samp:`loss, None`, where loss is a tensor of size N
                (per-element loss).
            should_normalize (bool) : If False, don't normalize the input
                (not recommended unless normalization is done in the
                custom_loss instead).
            orig_input (ch.tensor|None) : If not None, use this as the
                center of the perturbation set, rather than :samp:`x`.
            use_best (bool) : If True, use the best (in terms of loss)
                iterate of the attack process instead of just the last one.
            return_image (bool) : If True (default), then return the adversarial
                example as an image, otherwise return it in its parameterization
                (for example, the Fourier coefficients if 'constraint' is
                'fourier')
            est_grad (tuple|None) : If not None (default), then these are
                :samp:`(query_radius [R], num_queries [N])` to use for estimating the
                gradient instead of autograd. We use the spherical gradient
                estimator, shown below, along with antithetic sampling [#f1]_ 
                to reduce variance:
                :math:`\\\\nabla_x f(x) \\\\approx \\\\sum_{i=0}^N f(x + R\\\\cdot
                \\\\vec{\\\\delta_i})\\\\cdot \\\\vec{\\\\delta_i}`, where
                :math:`\\delta_i` are randomly sampled from the unit ball.
            mixed_precision (bool) : if True, use mixed-precision calculations
                to compute the adversarial examples / do the inference.
        Returns:
            An adversarial example for x (i.e. within a feasible set
            determined by `eps` and `constraint`, but classified as:

            * `target` (if `targeted == True`)
            *  not `target` (if `targeted == False`)

        .. [#f1] This means that we actually draw :math:`N/2` random vectors
            from the unit ball, and then use :math:`\\delta_{N/2+i} =
            -\\delta_{i}`.
        """
        # Can provide a different input to make the feasible set around
        # instead of the initial point
        if orig_input is None: orig_input = x.detach()
        orig_input = orig_input.cuda()

        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if targeted else 1

        # Initialize step class and attacker criterion
        criterion = ch.nn.CrossEntropyLoss(reduction='none')
        step_class = STEPS[constraint] if isinstance(constraint, str) else constraint
        step = step_class(eps=eps, orig_input=orig_input, step_size=step_size) 

        def calc_loss(inp, target):
            '''
            Calculates the loss of an input with respect to target labels
            Uses custom loss (if provided) otherwise the criterion
            '''
            if should_normalize:
                inp = self.normalize(inp)

            if custom_loss:
                return custom_loss(self.model, inp, target)
            else:
                output = self.model(inp)
                return criterion(output, target), output

        # Main attack loop
        if random_restarts:
            to_ret = None
            orig_cpy = x.clone().detach()
            for _ in range(random_restarts):
                x = orig_cpy.clone().detach()
                x = self.forward(x, target, constraint=constraint, eps=eps,
                    step_size=step_size, iterations=iterations, random_start=True,
                    random_restarts=0, do_tqdm=do_tqdm, targeted=targeted,
                    custom_loss=custom_loss, should_normalize=should_normalize,
                    orig_input=orig_input, use_best=use_best, return_image=return_image,
                    est_grad=est_grad, mixed_precision=mixed_precision)
                
                if to_ret is None:
                    to_ret = x
                else:
                    _, output = calc_loss(x, target)
                    _, best_output = calc_loss(to_ret, target)
                    replace = m * best_output < m * output
                    to_ret = ch.where(replace.view(-1, *([1] * (len(x.shape) - 1))), x, to_ret)
            
            return to_ret

        if random_start:
            x = step.random_perturb(x)

        iterator = range(iterations)
        if do_tqdm: iterator = tqdm(iterator)

        # Keep track of the "best" (worst-case) loss and its
        # corresponding input
        best_loss = None
        best_x = None

        # A function that updates the best loss and best input
        def replace_best(loss, bloss, x, bx):
            if bloss is None:
                bx = x.clone().detach()
                bloss = loss.clone().detach()
            else:
                replace = m * bloss < m * loss
                bx[replace] = x[replace].clone().detach()
                bloss[replace] = loss[replace]

            return bloss, bx

        # PGD iterates
        for _ in iterator:
            x = x.clone().detach().requires_grad_(True)
            losses, out = calc_loss(step.to_image(x), target)
            assert losses.shape[0] == x.shape[0], \
                    'Shape of losses must match input!'

            loss = ch.mean(losses)

            if step.use_grad:
                if (est_grad is None) and mixed_precision:
                    # Use torch.amp GradScaler for mixed precision
                    scaler = ch.amp.GradScaler("cuda")
                    with ch.amp.autocast("cuda"):
                        pass  # Forward pass already done
                    # Scale and backward
                    scaler.scale(loss).backward()
                    grad = x.grad.detach()
                    x.grad.zero_()
                elif (est_grad is None):
                    grad, = ch.autograd.grad(m * loss, [x])
                else:
                    f = lambda _x, _y: m * calc_loss(step.to_image(_x), _y)[0]
                    grad = helpers.calc_est_grad(f, x, target, *est_grad)
            else:
                grad = None

            with ch.no_grad():
                args = [losses, best_loss, x, best_x]
                best_loss, best_x = replace_best(*args) if use_best else (losses, x)

                x = step.step(x, grad)
                x = step.project(x)
                if do_tqdm: iterator.set_description("Current loss: {l}".format(l=loss))

        # Save computation (don't compute last loss) if not use_best
        if not use_best: 
            ret = x.clone().detach()
            return step.to_image(ret) if return_image else ret

        losses, _ = calc_loss(step.to_image(x), target)
        args = [losses, best_loss, x, best_x]
        best_loss, best_x = replace_best(*args)
        return step.to_image(best_x) if return_image else best_x


class AttackerModel(ch.nn.Module):
    def __init__(self, model, dataset, **kwargs):
        """
        Wrapper for any PyTorch model as an AttackerModel, which is 
        necessary for adversarial training and evaluation. 

        Args:
            model (nn.Module) : the model to wrap
            dataset : the dataset that the model is trained on, only used to get
            mean and std for data normalization
        """
        super(AttackerModel, self).__init__()
        self.model = model
        self.attacker = Attacker(model, dataset)

    def forward(self, inp, target=None, make_adv=False, with_latent=False,
                fake_relu=False, no_relu=False, with_image=True, **attacker_kwargs):
        """
        Args:
            inp : input to do inference on
            target (ch.tensor) : the labels of the input, used to make adversarial
                examples if desired
            make_adv (bool) : whether to make the input adversarial 
            with_latent (bool) : also return the second-to-last layer along
                with the logits
            fake_relu (bool) : replace ReLUs with incrementally smoothed
               versions for compatibility with TRADES and other defenses
            no_relu (bool) : remove all ReLUs for compatibility with TRADES
                and other defenses (see `here 
                <https://github.com/yaodongyu/TRADES/blob/master/models/small_cnn.py>`_
                for an example of how to do this)
            with_image (bool) : also return the image that was classified in
                addition to the logits
        Returns:
            logits : a tensor of shape [N, num_classes] with the logits from
            the model.
            latent_representation : (if with_latent=True) a tensor of 
            shape [N, D] where D is the second-to-last layer width.
            inp_adv : (if with_image=True) the image that was classified.
        """

        if make_adv:
            inp = self.attacker(inp, target, **attacker_kwargs)

        output = self.model(inp)

        if with_latent:
            return output, self.model.latent, inp if with_image else None
        else:
            return (output, inp) if with_image else output
