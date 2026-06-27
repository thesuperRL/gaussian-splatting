import torch
from process_image import tensorify, assign_coordinates
from parameter_init import initialize_parameters
from covariance import build_sigma
import matplotlib.pyplot as plt

def gaussian_density(coords, pos, scales, radians):
    flat_coord = torch.flatten(coords, end_dim=1).unsqueeze(1)
    x_min_mu = flat_coord - pos.unsqueeze(0)

    sigma_inv = build_sigma(scales, radians)

    mul = torch.einsum('pni,nij,pnj->pn', x_min_mu, sigma_inv, x_min_mu)

    G = torch.exp(-0.5 * mul)

    return G

def alpha_layering_render(G, colors, alpha, shape):
    opacity = torch.sigmoid(alpha)

    strength = G * opacity
    assert(strength.max().item() < 1 and strength.min().item() >= 0)

    order = torch.argsort(opacity, descending=True)
    strength = strength[:, order]
    sort_col = colors[order]

    invstr = 1 - strength
    scan = torch.cumprod(invstr, dim=1)
    ones_col = torch.ones(shape[0] * shape[1], 1)
    transmittance = torch.cat((ones_col, scan[:, :-1]), dim=1)
    assert(torch.allclose(transmittance[:, 0], torch.ones(len(transmittance[:, 0]))))

    weight = strength * transmittance

    render = torch.einsum('pn,nc->pc', weight, sort_col)
    render = render.reshape(shape).clamp(0, 1)
    return render


if __name__ == "__main__":
    image, shape = tensorify("images/shuak.png", resize = [256, 256])

    coords = assign_coordinates(image)

    pos, scale, colors, alpha, rot = initialize_parameters(image, coords)
    
    G = gaussian_density(coords, pos, scale, rot)

    # peek = G.max(dim=1).values.reshape(shape[:2])
    # plt.imshow(peek.detach().numpy(), cmap='gray')
    # plt.show()

    render = alpha_layering_render(G, colors, alpha, shape)
    plt.imshow(render.detach().numpy())
    plt.show()

