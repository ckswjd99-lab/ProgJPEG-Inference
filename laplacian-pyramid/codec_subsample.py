import cv2
import numpy as np
import os
import heapq
from collections import defaultdict
import time

def prod(val):
    val = list(val)
    res = 1
    for ele in val: 
        res *= ele 
    return res

def subsampling_encode(image, levels):
    """
    Encodes an image into a subsampling pyramid.

    Args:
        image (numpy.ndarray): Input image.
        levels (int): Number of levels in the pyramid.

    Returns:
        list: Subsampling pyramid (list of numpy arrays).
    """
    subsampling_pyramid = [image.astype(np.int16)]
    for i in range(levels - 1):
        image = image[::2, ::2].astype(np.int16)
        subsampling_pyramid.append(image)

    return subsampling_pyramid

def subsampling_decode(subsampling_pyramid, target_level, original_shape):
    """
    Decodes a subsampling pyramid back into an image.

    Args:
        subsampling_pyramid (list): Subsampling pyramid (list of numpy arrays).
        target_level (int): The level to reconstruct to.
        original_shape (tuple): The original shape of the image.

    Returns:
        numpy.ndarray: Reconstructed image.
    """
    image = subsampling_pyramid[target_level]
    # Upsample until the image matches the original shape
    for i in range(target_level):
        image = np.repeat(image, 2, axis=0)
        image = np.repeat(image, 2, axis=1)

    return image[:original_shape[0], :original_shape[1], :original_shape[2]]

def build_huffman_tree(data):
    """
    Builds a Huffman tree for the given data.

    Args:
        data (numpy.ndarray): Input data.

    Returns:
        dict: Huffman codes for each value in the data.
    """
    frequency = defaultdict(int)
    for value in data.ravel():
        frequency[value] += 1

    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    huffman_codes = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))
    return {symbol: code for symbol, code in huffman_codes}

def huffman_encode(data, huffman_codes):
    """
    Encodes the data using Huffman codes.

    Args:
        data (numpy.ndarray): Input data.
        huffman_codes (dict): Huffman codes.

    Returns:
        tuple: Encoded data as binary and its original bitstring length.
    """
    bitstring = ''.join(huffman_codes[value] for value in data.ravel())
    bitstring_length = len(bitstring)
    encoded_data = int(bitstring, 2).to_bytes((bitstring_length + 7) // 8, byteorder='big')
    return encoded_data, bitstring_length

def huffman_decode(encoded_data, bitstring_length, huffman_codes, shape):
    """
    Decodes a Huffman encoded binary data back into an array.

    Args:
        encoded_data (bytes): Huffman encoded binary data.
        bitstring_length (int): Length of the original bitstring.
        huffman_codes (dict): Huffman codes.
        shape (tuple): Shape of the original data.

    Returns:
        numpy.ndarray: Decoded data.
    """
    reverse_codes = {code: symbol for symbol, code in huffman_codes.items()}
    bitstring = bin(int.from_bytes(encoded_data, byteorder='big'))[2:]
    bitstring = bitstring.zfill(bitstring_length)

    decoded_data = []
    current_code = ""
    for bit in bitstring:
        current_code += bit
        if current_code in reverse_codes:
            decoded_data.append(reverse_codes[current_code])
            current_code = ""
    return np.array(decoded_data, dtype=np.int16).reshape(shape)

# Example usage:
if __name__ == "__main__":
    # Load an image
    image = cv2.imread("original.jpg")
    image = cv2.resize(image, (512, 512))  # Resize for simplicity
    original_shape = image.shape
    print(f"Original image {original_shape}: {prod(original_shape):,d} Bytes with {image.dtype}")

    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)

    # Encode image into Subsampling pyramid
    levels = 4
    start_time = time.time()
    subsampling_pyramid = subsampling_encode(image, levels)
    print(f"Creating Subsampling Pyramid: {time.time() - start_time:.4f} sec")
    print([x.shape for x in subsampling_pyramid])

    # Save intermediate Subsampling layers
    total_compressed_size = 0
    for level, subsampled in reversed(list(enumerate(subsampling_pyramid))):
        cv2.imwrite(f"output/Subsampling_Level_{level}.png", subsampled.astype(np.uint8))

        # Huffman encode each Subsampling layer
        start_time = time.time()
        huffman_codes = build_huffman_tree(subsampled)
        encoded_data, bitstring_length = huffman_encode(subsampled, huffman_codes)
        print(f"Encoding Subsampling Pyramid Level {level}: {time.time() - start_time:.4f} sec")

        # Calculate and print compressed size
        compressed_size = len(encoded_data)
        total_compressed_size += compressed_size

        uncompressed_size = prod(subsampled.shape)
        print(f" >> Compressed size {subsampled.shape}, {subsampled.dtype}: {compressed_size:,d} bytes (uncomp. {uncompressed_size:,d} bytes, {compressed_size / uncompressed_size * 100:.2f}%)")

        # Decode the encoded data for verification
        decoded_layer = huffman_decode(encoded_data, bitstring_length, huffman_codes, subsampled.shape)
        assert np.array_equal(decoded_layer, subsampled), "Decoded layer does not match original!"

    print(f"Total compressed size: {total_compressed_size:,d} bytes")

    # Decode Subsampling pyramid back to image
    reconstructed_image = subsampling_decode(subsampling_pyramid, 0, original_shape)

    # Save original and reconstructed images
    cv2.imwrite("output/Original_Image.png", image)
    cv2.imwrite("output/Reconstructed_Image.png", reconstructed_image)

    # Verify that the reconstructed image matches the original
    assert np.allclose(reconstructed_image, image), "Reconstructed image does not match original!"
