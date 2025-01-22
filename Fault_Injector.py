import random
import numpy as np
import time

def float_to_binary(value):
    """ Convert float to binary representation """
    import struct
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return f'{d:064b}'

def binary_to_float(b):
    """ Convert binary representation to float """
    import struct
    h = int(b, 2).to_bytes(8, byteorder="big")
    return struct.unpack(">d", h)[0]

def flip_bits(binary_str, positions):
    """ Flip the bits at the given positions in the binary string """
    b_list = list(binary_str)
    original_bits = [(pos, b_list[pos]) for pos in positions]
    for pos in positions:
        b_list[pos] = '1' if b_list[pos] == '0' else '0'
    flipped_bits = [(pos, b_list[pos]) for pos in positions]
    return ''.join(b_list), original_bits, flipped_bits
def zero_byte(binary_str, byte_position):
    """Zero out a byte at the specified byte position in the binary string"""
    start = byte_position * 8
    end = start + 8
    return binary_str[:start] + '0' * 8 + binary_str[end:]

def flip_msb(binary_str):
    """Flip the Most Significant Bit (MSB) of the binary string"""
    return flip_bits(binary_str, [0])

def flip_lsb(binary_str):
    """Flip the Least Significant Bit (LSB) of the binary string"""
    return flip_bits(binary_str, [63])

def inject_fault(model, fault_type, fault_model, bit_positions):
    start_time = time.time()  # Start time marker
    layers_with_params = []
    if fault_type == 'weight':
        layers_with_params = [layer for layer in model.layers if hasattr(layer, 'kernel')]
    elif fault_type == 'bias':
        layers_with_params = [layer for layer in model.layers if hasattr(layer, 'bias')]
    
    selected_layer = random.choice(layers_with_params)
    
    if fault_type == 'weight':
        param = selected_layer.kernel.numpy()
    elif fault_type == 'bias':
        param = selected_layer.bias.numpy()
    
    # Select a random parameter to flip bits
    param_size = param.size
    param_index = random.randint(0, param_size - 1)
    param_value = param.flat[param_index]
    
    # Convert to binary and inject the fault
    original_binary = float_to_binary(param_value)
    if fault_model == "1":
        # Single bit flip
        bit_positions = [random.randint(0, 63)]
        faulty_binary, original_bits, flipped_bits = flip_bits(original_binary, bit_positions)
    elif fault_model == "2":
        # Double bit flip
        bit_positions = random.sample(range(64), 2)
        faulty_binary, original_bits, flipped_bits = flip_bits(original_binary, bit_positions)
    elif fault_model == "3":
        # Byte zeroing (zero out one random byte)
        byte_position = random.randint(0, 7)
        faulty_binary = zero_byte(original_binary, byte_position)
        original_bits = [(i, original_binary[i]) for i in range(byte_position*8, (byte_position+1)*8)]
        flipped_bits = [(i, '0') for i in range(byte_position*8, (byte_position+1)*8)]
    elif fault_model == "4":
        # Most Significant Bit flip
        faulty_binary, original_bits, flipped_bits = flip_msb(original_binary)
    elif fault_model == "5":
        # Least Significant Bit flip
        faulty_binary, original_bits, flipped_bits = flip_lsb(original_binary)
    
    faulty_value = binary_to_float(faulty_binary)
    
    # Assign the faulty value back to the parameter
    if fault_type == 'weight':
        param.flat[param_index] = faulty_value
        selected_layer.kernel.assign(param)
    elif fault_type == 'bias':
        param[param_index] = faulty_value
        selected_layer.bias.assign(param)

    end_time = time.time()  # End time marker

    # Print the results
    print(f"Selected Layer: {selected_layer.name}")
    print(f"Fault Injected on: {fault_type}")
    print(f"Original Value: {param_value}")
    print(f"Faulty Value: {faulty_value}")
    print(f"Bit Positions: {bit_positions}")
    print(f"Original Bits: {original_bits}")
    print(f"Flipped Bits: {flipped_bits}")
    print(f"Fault injection took {end_time - start_time:.6f} seconds")