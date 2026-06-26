import torch
import torchvision.transforms.functional as func
import torchvision
import matplotlib.pyplot as plt

DEVICE = torch.device("cpu")

def tensorify(image_path, resize=None):
    raw_image = torchvision.io.decode_image(image_path, "RGB")
    float_image = (raw_image.to(torch.float32)) / 255.0
    if resize is not None:
        float_image = func.resize(float_image, resize)
    permuted_image = float_image.permute(1, 2, 0)
    final = permuted_image.to(DEVICE)
    return final, final.shape

def assign_coordinates(tensor_img, bounds = [-1, 1]):
    shape = list(tensor_img.shape) # results in (H, W, 3)

    x = torch.linspace(bounds[0], bounds[1], shape[1])
    y = torch.linspace(bounds[0], bounds[1], shape[0])

    enum_x, enum_y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.dstack([enum_x, enum_y])

    # coordinates match image pixels
    assert(coords.shape[:2] == tensor_img.shape[:2])
    assert(list(coords[0, 0]) == [-1.0, -1.0])
    assert(list(coords[-1, -1]) == [1.0, 1.0])

    return coords

if __name__ == "__main__":
    image, shape = tensorify("images/shuak.png", resize = [256, 256])

    print(image)
    print(shape)
    print(image.min())
    print(image.max())

    plt.imshow(image)
    plt.show()

    coords = assign_coordinates(image)
    print(coords)