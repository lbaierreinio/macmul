import random
import numpy as np

def ru_rowhammer(params):
    # Select
    random_index = np.random.choice(len(params))
    random_param = params[random_index]
    numpy_random_param = random_param.asnumpy()
    indices = tuple(np.random.randint(numpy_random_param.shape[i]) for i in range(numpy_random_param.ndim))
    num_float = numpy_random_param[indices]

    # Flip
    flipped_float = ru_flip(num_float)

    # Update
    numpy_random_param[indices] = flipped_float
    params[random_index].copyfrom(numpy_random_param)
    
    return params

def ru_flip(float_num):
    if not isinstance(float_num, np.float32):
        raise ValueError("Input must be a np.float32 number.")

    # Convert float to its binary representation (32 bits)
    float_bytes = float_num.tobytes()
    
    # Convert bytes to a binary string
    binary_representation = ''.join(format(byte, '08b') for byte in float_bytes)
    
    # Get the number of bits (should be 32 for np.float32)
    num_bits = len(binary_representation)
    
    # Randomly select a bit position to flip
    bit_position = random.randint(0, num_bits - 1)
    
    # Flip the specified bit
    bit_list = list(binary_representation)  # Convert string to a list for mutability
    bit_list[bit_position] = '1' if bit_list[bit_position] == '0' else '0'  # Flip the bit
    flipped_binary = ''.join(bit_list)
    
    # Convert binary string back to bytes
    flipped_bytes = int(flipped_binary, 2).to_bytes(4, byteorder='big')  # 4 bytes for np.float32
    
    # Convert bytes back to np.float32
    flipped_float = np.frombuffer(flipped_bytes, dtype=np.float32)[0]
    
    return flipped_float