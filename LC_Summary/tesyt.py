import numpy as np

# Example flattened list (replace this with your actual flattened list)
flattened_list = [i for i in range(1083437)]  # Replace this with your actual flattened list

# Check the total number of elements in the list
total_elements = len(flattened_list)
print("Total elements in the flattened list:", total_elements)

# Calculate the expected total elements for a 2D array with shape (34967, 31)
expected_elements = 34967 * 31
print("Expected elements for a (34967, 31) array:", expected_elements)

# Check if the total elements match the expected elements
if total_elements != expected_elements:
    print("The total elements in the list do not match the expected elements for a (34967, 31) array.")
else:
    # Reshape the flattened list into a 2D NumPy array with 34967 rows and 31 columns
    reshaped_array = np.array(flattened_list).reshape(34967, 31)
    print("Shape of the reshaped array:", reshaped_array.shape)