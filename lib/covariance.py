import torch
from process_image import tensorify, assign_coordinates
from parameter_init import initialize_parameters

def build_sigma(scale, radians):
    # activate scales and radians
    exp_scale = torch.exp(scale)
    s = torch.diag_embed(exp_scale)

    cos = torch.cos(radians)
    sin = torch.sin(radians)
    col1 = torch.stack((cos, sin), dim=1)
    col2 = torch.stack((-sin, cos), dim=1) 
    r = torch.stack((col1, col2), dim=2)

    sigma = r @ s @ s.transpose(-1, -2) @ r.transpose(-1, -2)
    sigma_inv = torch.linalg.inv(sigma)

    return sigma_inv

if __name__ == "__main__":
    image, shape = tensorify("images/shuak.png", resize = [256, 256])

    coords = assign_coordinates(image)

    _, scale, _, _, rot = initialize_parameters(image, coords)

    print(build_sigma(scale, rot))