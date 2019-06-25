import matplotlib.pyplot as plt
import numpy as np

def imshow(torch_pil_img, text=None, is_torch=False):

    plt.axis("off")

    if text:
        plt.text(75, 8, text, style="italic", fontweight="bold",
                 bbox={"facecolor": "white", "alpha": 0.8, "pad": 10})

    if is_torch:
        img = torch_pil_img.numpy()
        plt.imshow(np.transpose(img, (1, 2, 0)))
    else:
        img = np.array(torch_pil_img)
        plt.imshow(img)
    plt.show()

def plot_loss(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()
