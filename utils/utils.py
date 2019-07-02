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

def plot_history(history, filename=None):
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(121)
    ax.plot(history["epoch"], history["train_loss"], label="train loss")
    ax.plot(history["epoch"], history["val_loss"], label="val loss")
    ax = fig.add_subplot(122)
    ax.semilogy(history["epoch"], history["train_loss"], label="train loss")
    ax.semilogy(history["epoch"], history["val_loss"], label="val loss")
    
    if filename is not None:
        plt.savefig(filename)
    
    plt.show()
    
    