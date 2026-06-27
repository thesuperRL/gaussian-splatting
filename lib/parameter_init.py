import torch
from process_image import tensorify, assign_coordinates
import matplotlib.pyplot as plt
import torch.nn as nn
from bisect import bisect_left
from math import log

GAUSSIAN_AMT = 500
DEVICE = "cpu"

def binary_search_2d(item, body_x, body_y):
    # know that we are passing in coords as body which are naturally ordered, safe to binsearch one direction and then binsearch another
    x = bisect_left(body_x, item[0])
    y = bisect_left(body_y, item[1])
    return x, y

def initialize_parameters(tensor_img, coords, N = GAUSSIAN_AMT):
    positions = torch.empty(N, 2)
    positions = positions.uniform_(-1, 1)
    param_pos = nn.Parameter(positions)

    scales = torch.empty(N, 2)
    scales = scales.uniform_(log(0.001), log(0.1))
    param_sca = nn.Parameter(scales)

    colors = torch.empty(N, 3)
    coords_list = torch.unbind(coords, dim=2)
    x_list = list(coords_list[0][0])
    y_list = list(coords_list[1][:, 0])
    for index, pos in enumerate(positions):
        x, y = binary_search_2d(pos, x_list, y_list)
        colors[index] = tensor_img[y, x]
    param_col = nn.Parameter(colors)

    alpha = torch.zeros(N, )
    param_alp = nn.Parameter(alpha)

    radians = torch.zeros(N, )
    radians = radians.uniform_(0, 2 * torch.pi)
    param_rad = nn.Parameter(radians)

    return param_pos, param_sca, param_col, param_alp, param_rad

if __name__ == "__main__":
    image, shape = tensorify("images/shuak.png", resize = [256, 256])

    coords = assign_coordinates(image)

    param_tup = initialize_parameters(image, coords)
    for param in param_tup:
        print(param.shape)
        print(param.requires_grad)
        print(param.min())
        print(param.max())