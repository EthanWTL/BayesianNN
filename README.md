# Bayesian Theorem's application in Math and Machine Leanring Field.

![Contributors](https://img.shields.io/github/contributors/EthanWTL/BayesianNN?style=plastic)
![Stars](https://img.shields.io/github/stars/EthanWTL/BayesianNN)
![Licence](https://img.shields.io/github/license/EthanWTL/BayesianNN)
![Issues](https://img.shields.io/github/issues/EthanWTL/BayesianNN)

## Introduction:
Welcome,

This is a repo associate with the ongoing paper [A Survey about Bayesian Inference Power in Math and Machine Learning Perspectives](Bayesian_Inference_first_draft.pdf)

Besides explaining topics included in the paper, this repo will focus on walking through the implementation of [MCMC](HMC_winequality.ipynb) and [Variational Inference](HMC_winequality.ipynb) in the Linear Regression field.

## Intuition:
After examine the power of Bayesian theorem in math and ML field, It is impressive to see the collaborated work done by different teams for decades in physics, math and ML fields that keep upgrading methods in MCMC and VI.

But there is lack of research that compiling all methods on the track together -- a comprehensive history of the evolution of Bayesian Theorem. Thus we have this paper that aim at conducting a detailed math analysis of each method, plugins and upgraded for Bayesian Inference.


## Background:
Here are some basic formula that we need for future experiment.

[Joint Probability Theorem]():

$$  P(A\mid B) = \frac{\lvert A \cap B \rvert}{\lvert B \rvert}=\frac{\dfrac{\lvert A \cap B \rvert}{\lvert \Omega \rvert}}{\dfrac{\lvert B \rvert}{\lvert \Omega \rvert}} = \frac {P(A \cap B)}{P(B)}  $$

[Bayesian Theorem](https://philpapers.org/rec/SWIBT-2):

$$ P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)} $$


