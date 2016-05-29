# Short Term
* Clean up old code
* Create CD / poster / report

# Medium term
* Get old code to same standards of documentation as new code  

# Longer Term
* Tile based experimentation
  * Reinforcement Learning and Dynamic Programming Using Function Approximators
  * Does linear function approximation really work well enough?
  * How fast / accurate is this compared to previous work?
* Fixing and understanding older work
  * Open loop debugging (esp. for dynamic programming with neural interpolation)
  * For NFQ, look up dynamic networks (how to incorporate newer information properly)
  * For table based method:
    * Make sure deep copying in learningCycle() in tableBasedMethods.py is done properly
    * Make sure neural network is training properly on table (why doesn't it match that well?)