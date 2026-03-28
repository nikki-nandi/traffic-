def predict(state):
    return 1 if (state[0] + state[1]) % 10 == 0 else 0