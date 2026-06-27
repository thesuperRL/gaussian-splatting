import torch
from process_image import tensorify, assign_coordinates
from parameter_init import initialize_parameters
from covariance import build_sigma
from visualizer import gaussian_density, alpha_layering_render
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def optimize(img_path, size = [256, 256], splats = 1000, iter = 1000, verbose = False, showper = 20, store_each = False):
    dir = f"output/shu_{size[0]}_{size[1]}_{iter}"
    Path(dir).mkdir(parents=True, exist_ok=True)
    
    image, shape = tensorify(img_path, resize = size)
    coords = assign_coordinates(image)
    pos, scale, colors, alpha, rot = initialize_parameters(image, coords, N = splats)
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

        if verbose:
            print(i)
            print(loss.item())
        if showper and i % showper == 0:
            img_window.set_data(render.detach().numpy())
            plt.pause(0.0001)
        if store_each:
            filename = dir + f"/shu_{i}.png"
            plt.imsave(filename, render.detach().numpy())
    
    filename = dir+"/final.png"
    plt.imsave(filename, render.detach().numpy())
    plt.ioff()
    plt.show()

    return dir

def label_frame(path, text):
    im = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        draw.text((5 + dx, 5 + dy), text, fill=(0, 0, 0), font=font)
    draw.text((5, 5), text, fill=(255, 255, 255), font=font)
    return im

def save_as_gif(dir, iter, base_ms = 50, end_ms = 1000):
    frames = [label_frame(dir + f"/shu_{i}.png", str(i)) for i in range(iter)]
    frames.append(label_frame(dir + "/final.png", "final"))

    durations = [base_ms] * iter
    durations.append(end_ms)

    frames[0].save(
        "output/gifs/shu.gif",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0
    )

if __name__ == "__main__":
    iterations = 1000

    dir = "/Users/ryanli/Documents/GitHub/gaussian-splatting/output/shu_256_256_1000"
    #dir = optimize("images/shuak.png", size = [256, 256], splats = 2000, iter = iterations, showper = 20, store_each = True)

    save_as_gif(dir, iterations)