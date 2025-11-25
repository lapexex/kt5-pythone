import numpy as np
import arrayprocessor as ap

def test_process_inplace():
    print("Testing process_inplace...")
    original = np.array([0.0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2], dtype=np.float64)
    test_array = original.copy()
    print("Original array:", original)
    
    ap.process_inplace(test_array)
    print("Processed array:", test_array)
    
    expected = np.sin(original) * np.cos(original)
    print("Expected result:", expected)
    
    if np.allclose(test_array, expected):
        print("✓ Test passed.")
    else:
        print("✗ Test failed!")
    print("-" * 50)

def test_create_processed():
    print("Testing create_processed...")
    original = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    print("Original array:", original)
    
    result = ap.create_processed(original)
    print("Processed array:", result)
    
    expected = (original + 1) ** 2
    print("Expected result:", expected)
    
    if np.allclose(result, expected):
        print("✓ Test passed.")
    else:
        print("✗ Test failed!")
    print("-" * 50)

def test_matrix_multiply():
    print("Testing matrix_multiply...")
    matrix1 = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=np.float64)
    matrix2 = np.array([[7.0, 8.0],
                       [9.0, 10.0],
                       [11.0, 12.0]], dtype=np.float64)
    print("Matrix 1:")
    print(matrix1)
    print("Matrix 2:")
    print(matrix2)
    
    result = ap.matrix_multiply(matrix1, matrix2)
    print("Result from C++:")
    print(result)
    
    expected = np.dot(matrix1, matrix2)
    print("Expected result (NumPy):")
    print(expected)
    
    if np.allclose(result, expected):
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")
    print("-" * 50)

def benchmark_comparison():
    print("Performance benchmark...")
    import time
    
    size = 1000000
    test_array = np.random.random(size).astype(np.float64)
    
    start_time = time.time()
    python_result = np.sin(test_array) * np.cos(test_array)
    python_time = time.time() - start_time
    
    test_array_copy = test_array.copy()
    start_time = time.time()
    ap.process_inplace(test_array_copy)
    cpp_time = time.time() - start_time
    
    print(f"Array size: {size}")
    print(f"Python time: {python_time:.6f} seconds")
    print(f"C++ time: {cpp_time:.6f} seconds")
    print(f"Speedup: {python_time/cpp_time:.2f}x")
    
    if np.allclose(python_result, test_array_copy):
        print("✓ Results are identical")
    else:
        print("✗ Results differ!")

if __name__ == "__main__":
    print("=" * 50)
    print("ARRAY PROCESSOR MODULE TESTS")
    print("=" * 50)
    
    test_process_inplace()
    test_create_processed()
    test_matrix_multiply()
    benchmark_comparison()
    
    print("=" * 50)
    print("All tests completed!")