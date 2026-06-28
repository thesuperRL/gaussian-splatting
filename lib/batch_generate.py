from optimization import optimize, save_as_gif
from data_loader import get_operator_name, get_random_icons_as_strings

if __name__ == "__main__":
    for img in get_random_icons_as_strings(100):
        print("Processing: " + img)
        iterations = 1000
        storeper = 100
        dir = optimize(img, get_operator_name(img), size = [256, 256], splats = 1000, iter = iterations, showper = None, storeper = storeper, store_each = True, store_tensor = True)
        save_as_gif(dir, get_operator_name(img), iterations, showper=storeper)