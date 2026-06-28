import torch
from process_image import tensorify, assign_coordinates
from parameter_init import initialize_parameters
from covariance import build_sigma
from visualizer import gaussian_density, alpha_layering_render
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def optimize(img_path, name, size = [256, 256], splats = 1000, iter = 1000, verbose = False, showper = 20, storeper = 20, store_each = False, store_tensor = False):
    dir = f"output/{name}_{size[0]}_{size[1]}_{iter}"
    Path(dir).mkdir(parents=True, exist_ok=True)
    
    image, shape = tensorify(img_path, resize = size)
    coords = assign_coordinates(image)
    pos, scale, colors, alpha, rot = initialize_parameters(image, coords, N = splats)
    params = [pos, scale, colors, alpha, rot]

    optimizer = torch.optim.Adam(params, lr = 1e-3)
    target = image

    if showper:
        plt.ion()
        img_window = plt.imshow(image)

    for i in range(iter):
        optimizer.zero_grad()
        G = gaussian_density(coords, pos, scale, rot)
        render = alpha_layering_render(G, colors, alpha, shape)

        loss = torch.mean((render - target) ** 2)
        scale_penalty = torch.mean((scale[:, 0] - scale[:, 1])**2)
        total_loss = loss + 0.01 * scale_penalty
        total_loss.backward()
        optimizer.step()

        if verbose:
            print(i)
            print(loss.item())
        if showper and i % showper == 0:
            print(f"Iteration {i}")
            img_window.set_data(render.detach().numpy())
            plt.pause(0.0001)
        if store_each and i % storeper == 0:
            filename = dir + f"/{name}_{i}.png"
            plt.imsave(filename, render.detach().numpy())
    
    filename = dir + "/final.png"
    plt.imsave(filename, render.detach().numpy())
    plt.ioff()
    plt.show(block=False)

    if store_tensor:
        torch.save(render, dir + "/tensor.pt")

    return dir

def label_frame(path, text):
    im = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        draw.text((5 + dx, 5 + dy), text, fill=(0, 0, 0), font=font)
    draw.text((5, 5), text, fill=(255, 255, 255), font=font)
    return im

def save_as_gif(dir, name, iter, base_ms = 50, end_ms = 1000, showper = 100):
    frames = [label_frame(dir + f"/{name}_{i}.png", str(i)) for i in range(0, iter, showper)]
    frames.append(label_frame(dir + "/final.png", "final"))

    durations = [base_ms] * len(frames)
    durations[-1] = end_ms

    frames[0].save(
        f"output/gifs/{name}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0
    )

if __name__ == "__main__":
    iterations = 1000
    storeper = 20

    dir = optimize("images/shuak.png", "shu", size = [256, 256], splats = 2000, iter = iterations, showper = 20, storeper = storeper, store_each = True)

    save_as_gif(dir, "shu", iterations, showper=storeper)