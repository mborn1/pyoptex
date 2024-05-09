class RestartEveryNFailed:
    """
    Restarts every N rejected samples. This is to counter
    bad search regions and reset to the best previsouly found
    design (state).

    Attributes:
    i : int
        The current number of consecutively rejected iterations
    max_it : int
        The maximum number of consecutively rejected iterations
    """
    def __init__(self, max_it):
        self.i = 0
        self.max_it = max_it

    def reset(self):
        self.i = 0

    def accepted(self):
        self.i = 0

    def rejected(self):
        self.i += 1

    def call(self, state, best_state):
        if self.i > self.max_it:
            self.i = 0
            print('Restarted the optimization from optimum')
            return best_state
        else:
            return state
