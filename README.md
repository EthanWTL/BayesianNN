# Bayesian Theorem's application in Math and Machine Leanring Field.

![Contributors](https://img.shields.io/github/contributors/EthanWTL/BayesianNN?style=plastic)
![Stars](https://img.shields.io/github/stars/EthanWTL/BayesianNN)
![Licence](https://img.shields.io/github/license/EthanWTL/BayesianNN)
![Issues](https://img.shields.io/github/issues/EthanWTL/BayesianNN)

Welcome, :grinning:

This is a repo associate with the ongoing paper [A Survey about Bayesian Inference Power in Math and Machine Learning Perspectives](Bayesian_Inference_first_draft.pdf)

Besides explaining topics included in the paper, this repo will focus on walking through the implementation of [MCMC](HMC_winequality.ipynb) and [Variational Inference](HMC_winequality.ipynb) in the Linear Regression field.

## Intuition:
After examine the power of Bayesian theorem in math and ML field, It is impressive to see the collaborated work done by different teams for decades in physics, math and ML fields that keep upgrading methods in MCMC and VI.

But there is lack of research that compiling all methods on the track together -- a comprehensive history of the evolution of Bayesian Theorem. Thus we have this paper that aim at conducting a detailed math analysis of each method, plugins and upgraded for Bayesian Inference.

---
## Backgrounds review: 
### 1. Bayesian Theorem:

```Joint Probability Theorem:``` $P(A\mid B) = \frac{\lvert A \cap B \rvert}{\lvert B \rvert}= \frac {P(A \cap B)}{P(B)}$

```Bayesian Inference:``` $P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}$


### 2. Bayesian Neural Network:

```latent variables:``` $p(z\mid x) = \frac{p(x \mid z) p(z)}{p(x)}$

```General form:``` $p(x) = \frac{f(x)}{NC}$


### 3. MCMC:

 ```Markov Chain:``` Each variable $z$ is going to be one of the state in Markov Chain. The steady state is our target varible set.
 
 ```Hamiltonian Monte Carlo:``` Introducing physic idea position and momentum for better state candidate proposal.
 
 ```No-U-Turn-Sampler:``` eliminating the long term U-turn path, auto tune step size and time in leapfrog method.



### 4. Variational Inference:

```inference distribution: ``` find a "similar" distribution to intractable posterior $q_{\phi}(z \mid x) \approx p_{\theta} (z \mid x) \nonumber$

```KL-Divergence:```  used to determine the different between true posterior and inference distribution $D_{KL}(q_{\phi},p_{\theta})$

```Evidence Lower Bound:``` rearrange KL-Divergence to switch question into maximizing ELBO. $L_{\theta,\phi(x)} = E_{q_{\Phi} (z \mid x)} \left[log \, p_{\theta}(x,z) - log \, q_{\Phi}(z\mid x) \right]$

```Reparameterization trick:``` during backpropgation, introduce noise $\epsilon$, so $z = g(\phi, x, \epsilon)$ to apply gradient on ELBO

  ---


## Experiment:
Here we will walk through the experiement for [MCMC](HMC_winequality.ipynb) and [VI]().

### MCMC

Define Linear regression Logic:
```python
coeffs = ed.Normal(loc=tf.zeros([D,1]),scale=tf.ones([D,1]),name="coeffs")
bias = ed.Normal(loc=tf.zeros([1]), scale=tf.ones([1]),name="bias") 
noise_std = ed.HalfNormal(scale=tf.ones([1]),name="noise_std")

predictions = ed.Normal(loc=tf.matmul(features, coeffs)+bias,scale=noise_std,name="predictions")
```

Create Joint probability function for Markov Chain
   
```python
def target_log_prob_fn(coeffs, bias, noise_std):
  return ed.make_log_joint_fn(features=x, coeffs=coeffs, bias=bias, noise_std=noise_std, predictions=y)
```

Define a No-U-Turn-Sampler Kernel
```python
kernel = tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn=target_log_prob_fn,
    step_size=step_size)
```
