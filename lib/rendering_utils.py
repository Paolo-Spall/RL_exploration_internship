
import numpy as np


def fig_to_rgb(fig):
    # 1. Draw the canvas
    fig.canvas.draw()
    
    # 2. Convert the canvas to a buffer
    # Note: 'buffer_rgba' is usually the fastest
    img_array = np.array(fig.canvas.buffer_rgba())
    
    # 3. Slice off the Alpha channel (RGBA -> RGB)
    return img_array[:, :, :3]

