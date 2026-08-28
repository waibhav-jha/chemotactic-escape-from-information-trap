import numpy as np

class QLearner:
    def __init__(self,n_states,n_actions=2,alpha=.1,gamma=.99,
                 epsilon=1.,epsilon_min=.05,epsilon_decay=.995):
        self.Q=np.zeros((n_states,n_actions))
        self.alpha=alpha; self.gamma=gamma
        self.epsilon=epsilon; self.epsilon_min=epsilon_min
        self.epsilon_decay=epsilon_decay

    def choose_action(self,state,rng=None):
        rng=np.random.default_rng() if rng is None else rng
        if rng.random()<self.epsilon:
            return int(rng.integers(self.Q.shape[1]))
        return int(np.argmax(self.Q[state]))

    def update(self,state,action,reward,next_state,done):
        target=reward
        if not done:
            target += self.gamma*np.max(self.Q[next_state])
        self.Q[state,action] += self.alpha*(target-self.Q[state,action])

    def decay(self):
        self.epsilon=max(self.epsilon_min,self.epsilon*self.epsilon_decay)

if __name__=="__main__":
    q=QLearner(20)
    print("Q-table shape:",q.Q.shape)
