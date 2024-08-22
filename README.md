# Itera Development Board
### Genetic Algorithm-based optimization engine

This repo contains a minimal optimization engine currently developed for simple games.

The Itera framework consists of a **neural network** which uses a self-contained math library (for matrices operations) and then some methods that allows the user to train a set of networks.
The networks can be trained using every program that is able to provide a score.

Currently, an example has been developed in which the framework uses a **Game** interface, according to which a game must simply provide a **Play** method that returns a score so that the engine can perform evaluation on it.

If you run the demo, a **Pong** game is used as an environment for the networks to be trained. You will see the agent that moves the paddle gradually improve.

Feel free to start from this work to make cool stuff like other games or more sophisticated algorithms to train the networks.
