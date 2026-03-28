def predict(state):
    NS, EW, _, _, _ = state
    return 0 if NS > EW else 1