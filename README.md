# A2C From Scratch

Minimal Actor-Critic neural network in pure Java. Single hidden layer with tanh activation, softmax policy head, and linear value head. No dependencies.

## Architecture

```
inputs (D) --> [tanh hidden layer (H)] --> actor logits (A) --> softmax --> pi(a|s)
                        |
                        +--> value head --> V(s)
```

Shared trunk with separate heads for policy and value estimation.

## Algorithm

One-step TD(0) Actor-Critic with entropy regularization.

**TD Error:**
```
delta = r + gamma * V(s') - V(s)
```

**Critic update:**
```
w_critic += alpha_critic * delta * grad V(s)
```

**Actor update (policy gradient + entropy bonus):**
```
w_actor += alpha_actor * delta * grad log pi(a|s) + beta * grad H(pi)
```

The policy gradient for softmax logits:
```
d log pi(a) / d logit(k) = I[k==a] - pi(k)
```

## Features

- Xavier initialization for tanh activation
- Numerically stable softmax (max subtraction)
- TD error clipping for gradient stability
- Separate learning rate for trunk backpropagation
- Pre-allocated buffers to avoid per-step allocations
- Correct entropy gradient via softmax Jacobian

## API

```java
// Create network: 4 inputs, 16 hidden units, 2 actions
TinyActorCriticANN agent = new TinyActorCriticANN(4, 16, 2);

// Inference
int action = agent.sampleAction(state);
double[] probs = agent.policyProbs(state);
double value = agent.value(state);

// Learning
double delta = agent.update(
    state,          // current state
    action,         // action taken
    reward,         // observed reward
    nextState,      // next state
    terminal,       // episode ended?
    0.001,          // alpha actor
    0.01,           // alpha critic
    0.001,          // alpha trunk
    0.99            // gamma
);
```

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| ENTROPY_BETA | 0.005 | Entropy bonus coefficient, try 0.001 to 0.02 |
| TD clip range | [-5, 5] | Prevents large gradient updates |

## License

MIT
