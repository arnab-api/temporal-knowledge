# Context-dependent Fact Recall in LLMs
Some of the facts are context dependent; such as: the *president of a country* changes over time. Sometimes, the context has significant impact on how LLMs recalls facts, like

```
Who caused the 9/11 attacks? --> Al-Qaeda
Who really caused the 9/11 attacks? --> US Government
```
Or, 
```
A 4th grader is solving the problem [some math problem]. He gets the answer --> [Some wrong answer]

Prof Smith is solving the problem [some math problem]. He gets the answer --> [correct answer!]
```

In this project we will try to understand how LLMs implement context-dependent fact recall. We will primarity focus on temporal context, since recent LLaMA-2 models are kind of good at it.



## Setup

To run the code, create a virtual environment with the tool of your choice, e.g. conda:
```bash
conda create --name relations python=3.10
```
Then, after entering the environment, install the project dependencies:
```bash
python -m pip install invoke
invoke install
```