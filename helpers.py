import numpy as np

# Based on common color blindness
# https://www.nature.com/articles/nmeth.1618/figures/2
# Skip black and reserve it for other specific lines
def color_list():
    l = [
            np.array([230, 159, 0]), # orange
            np.array([86, 180, 233]), # Sky blue
            np.array([0, 158, 115]), # Bluish green
            np.array([240, 228, 66]), # Yellow
            np.array([0, 114, 178]), # Blue
            np.array([213, 94, 0]), # Vermillion
            np.array([204, 121, 167]), # Reddish purple
    ]
    return [i/255. for i in l]
