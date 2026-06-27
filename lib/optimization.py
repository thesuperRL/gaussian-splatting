import torch
from process_image import tensorify, assign_coordinates
from parameter_init import initialize_parameters
from covariance import build_sigma
from visualizer import gaussian_density, alpha_layering_render
import matplotlib.pyplot as plt

def optimize(img_path, size = [256, 256], iter = 1000, showper = 20):
    image, shape = tensorify(img_path, resize = size)
    coords = assign_coordinates(image)
    pos, scale, colors, alpha, rot = initialize_parameters(image, coords, N = 1000)
    params = [pos, scale, colors, alpha, rot]

    optimizer = torch.optim.Adam(params, lr = 1e-2)
    target = image

    if showper:
        plt.ion()
        img_window = plt.imshow(image)

    for i in range(iter):
        optimizer.zero_grad()
        G = gaussian_density(coords, pos, scale, rot)
        render = alpha_layering_render(G, colors, alpha, shape)

        loss = torch.mean((render - target) ** 2)
        loss.backward()
        optimizer.step()

        if showper and i % showper == 0:
            print(i)
            print(loss.item())
            img_window.set_data(render.detach().numpy())
            plt.pause(0.0001)
    
    filename = f"../output/shu_{size[0]}_{size[1]}_{iter}.png"
    plt.imsave(filename, render)
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    optimize("images/shuak.png", size = [256, 256], iter = 1000, showper = 20)