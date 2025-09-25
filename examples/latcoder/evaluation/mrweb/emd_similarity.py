from PIL import Image
import numpy as np
from lap import lapjv
from multiprocessing import Pool

def solve_assignment_lapjv(cost_matrix):
    _, col_idx, _ = lapjv(cost_matrix)
    return col_idx


def process_imgs(image1, image2, max_size):
    # Get the original sizes
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Determine the new dimensions (max of both images' width and height)
    new_width = max(width1, width2)
    new_height = max(height1, height2)

    # Pad images to the new dimensions with random values
    def pad_image(image, new_width, new_height):
        # Create a random padded background with the new dimensions
        random_padding = np.random.randint(0, 256, (new_height, new_width, 3), dtype=np.uint8)
        padded_image = Image.fromarray(random_padding)

        # Paste the original image onto the padded background (placing in the top-left corner)
        padded_image.paste(image, (0, 0))

        return padded_image

    padded_image1 = pad_image(image1, new_width, new_height)
    padded_image2 = pad_image(image2, new_width, new_height)

    # Calculate the aspect ratio for resizing to the max size
    aspect_ratio = min(max_size / new_width, max_size / new_height)
    new_size = (int(new_width * aspect_ratio), int(new_height * aspect_ratio))

    # Resize the padded images to the specified max size
    resized_image1 = padded_image1.resize(new_size, Image.LANCZOS)
    resized_image2 = padded_image2.resize(new_size, Image.LANCZOS)

    # resized_image1.show()
    # resized_image2.show()

    # Convert the images to numpy arrays with dtype int16
    array1 = np.array(resized_image1).astype(np.int16)
    array2 = np.array(resized_image2).astype(np.int16)

    return array1, array2



def calculate_emd_sim(img_array1, img_array2):
    """img_array1 is the original image, img_array2 is the generated image"""
    if len(img_array1.shape) == 2:
        flat_array1 = img_array1.flatten()
        flat_array2 = img_array2.flatten()

        cost_matrix = np.abs(flat_array1[:, None] - flat_array2[None, :])
        _, col_idx, _ = lapjv(cost_matrix)

        total_min_cost = cost_matrix[np.arange(len(flat_array1)), col_idx].sum()
        max_cost = np.maximum(flat_array1, 255 - flat_array1).sum()
        normalized_min_cost = total_min_cost / max_cost

    else:
        red1, green1, blue1 = img_array1[:, :, 0], img_array1[:, :, 1], img_array1[:, :, 2]
        red2, green2, blue2 = img_array2[:, :, 0], img_array2[:, :, 1], img_array2[:, :, 2]

        flat_red1, flat_green1, flat_blue1 = red1.flatten(), green1.flatten(), blue1.flatten()
        flat_red2, flat_green2, flat_blue2 = red2.flatten(), green2.flatten(), blue2.flatten()

        cost_matrix_red = np.abs(flat_red1[:, None] - flat_red2[None, :]).astype(np.float32)
        cost_matrix_green = np.abs(flat_green1[:, None] - flat_green2[None, :]).astype(np.float32)
        cost_matrix_blue = np.abs(flat_blue1[:, None] - flat_blue2[None, :]).astype(np.float32)

        with Pool(processes=3) as pool:
            results = pool.map(solve_assignment_lapjv, [cost_matrix_red, cost_matrix_green, cost_matrix_blue])
            col_ind_red = results[0]
            col_ind_green = results[1]
            col_ind_blue = results[2]
        
        min_cost_red_lapjv = cost_matrix_red[np.arange(len(flat_red1)), col_ind_red].sum()
        min_cost_green_lapjv = cost_matrix_green[np.arange(len(flat_green1)), col_ind_green].sum()
        min_cost_blue_lapjv = cost_matrix_blue[np.arange(len(flat_blue1)), col_ind_blue].sum()

        total_min_cost_lapjv = min_cost_red_lapjv + min_cost_green_lapjv + min_cost_blue_lapjv
        max_cost = np.maximum(flat_red1, 255 - flat_red1).sum() + np.maximum(flat_green1, 255 - flat_green1).sum() + np.maximum(flat_blue1, 255 - flat_blue1).sum()
        normalized_min_cost = total_min_cost_lapjv / max_cost

    # return {"cost": total_min_cost_lapjv, "normalized_sim": 1 - normalized_min_cost}
    return 1 - normalized_min_cost

def emd_similarity(img1: Image.Image, img2:Image.Image, max_size=96, mode="RGB"):
    """not symmetric, the first image is the original image, the score is normalized according to the original image"""
    image1 = img1.convert(mode)
    image2 = img2.convert(mode) 

    array1, array2 = process_imgs(image1, image2, max_size)
    similarity = calculate_emd_sim(array1, array2)

    return similarity
    