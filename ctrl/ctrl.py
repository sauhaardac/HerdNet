def initialize_state():
    """Randomly initialize state

    Returns:
    x (np.ndarray): the intial state of the system
    """

def generate_control_action(x, u_m):
    """Generate control action given state and motion primitive

    Parameters:
    x (np.ndarray): the current state of the system
    u_m (np.ndarray): the motion primitive

    Returns:
    u_c (np.ndarray): the control action for input into dynamics
    """

    pass

def simulate_dynamics(x, u_c):
    """Get the next state given the control action and current state

    Parameters:
    x (np.ndarray): the current state of the system
    u_c (np.ndarray): the control action

    Returns:
    x_next (np.ndarray): the updated state
    """

    pass
