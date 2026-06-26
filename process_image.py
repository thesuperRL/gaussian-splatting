import torch
import torchvision
import matplotlib.pyplot as plt

DEVICE = torch.device("cpu")

def tensorify(image_path):
    raw_image = torchvision.io.decode_image(image_path, "RGB")
    float_image = (raw_image.to(torch.float32)) / 255.0
    permuted_image = float_image.permute(1, 2, 0)
    permuted_image.to(DEVICE)
    return permuted_image, permuted_image.shape

if __name__ == "__main__":
    image, shape = tensorify("images/shuak.png")
    print(image)
    print(shape)
    plt.imshow(image)
    plt.show()