

class Simple_Train:
    def __init__(self):
        pass;





def choose_methodology(option):
    if "simple" in option: return Simple_Train()
    elif "other" in option: return None
    else: None
    return None