Vector Library - A Lua library for managing vectors 

The Vector Library is a Lua library that provides functions for managing and manipulating vectors. It can be used in a variety of applications, such as machine learning, computer graphics, physics simulations, and more. 

Usage case: 
Suppose you want to perform some operations on a set of vectors. You can use the Vector Library to create, manipulate, and analyze these vectors. Here are some example codes to illustrate how the Vector Library can be used: 

Example 1: Creating a vector 
local Vector = require("Vector")

-- create a vector
local v = Vector.create_vector({1, 2, 3})

-- print the vector
for i = 1, Vector.get_size(v) do
  print(v[i])
end

Output: 
1
2
3


Example 2: Scaling a vector 
local Vector = require("Vector")

-- create a vector
local v = Vector.create_vector({1, 2, 3})

-- scale the vector
local scaled_v = Vector.scaling_vectors(v, 2)

-- print the scaled vector
for i = 1, Vector.get_size(scaled_v) do
  print(scaled_v[i])
end

Output: 
2
4
6

Example 3: Normalizing a vector 
local Vector = require("Vector")

-- create a vector
local v = Vector.create_vector({3, 4})

-- normalize the vector
local normalized_v = Vector.norm_vectors(v)

-- print the normalized vector
for i = 1, Vector.get_size(normalized_v) do
  print(normalized_v[i])
end

Output: 
0.6
0.8
The Vector Library is a Lua library that provides various functions for manipulating and performing operations on vectors. A vector is a mathematical object that represents a quantity with both magnitude and direction. In computer science, vectors are commonly used to represent data that has multiple dimensions.

Here's a full breakdown of the functions provided by the Vector Library:
Vector.check_boolean(boolean_string, vector)
This function takes in a boolean string and a vector table, and checks if the boolean is true for the given vector. If the boolean is false, an error message is returned. If the boolean is true and the vector has a value, the vector is returned. If the boolean is true and the vector is nil, an error message is returned.
Vector.create_vector(vector_values)
This function takes in a table of values and creates a vector from those values. The resulting vector is returned as a table.
Vector.get_size(vector)
This function takes in a vector table and returns the number of elements in the vector.
Vector.save_vector(vector)
This function takes in a vector table and saves it to a file named "Vector_Var.txt". If the file does not exist, it will be created. The vector is appended to the end of the file as a table.
Vector.add_vectors()
This function reads all the vectors stored in "Vector_Var.txt" and returns a new vector that is the sum of all the vectors.
Vector.sub_vectors()
This function reads all the vectors stored in "Vector_Var.txt" and returns a new vector that is the difference of all the vectors.
Vector.scaling_vectors(scaling_factor)
This function reads a vector from "Vector_Var.txt" and scales it by a given factor. The resulting vector is returned as a table.
Vector.rotate_vector(rotation_angle)
This function reads a vector from "Vector_Var.txt" and rotates it by a given angle. The resulting vector is returned as a table.
Vector.norm_vectors()
This function reads a vector from "Vector_Var.txt" and normalizes it. The resulting vector is returned as a table.
Vector.project_vectors(target_vector)
This function reads a vector from "Vector_Var.txt" and projects it onto a target vector. The resulting projected vector is returned as a table.
Vector.dimreduct_vectors(target_dimension)
This function reads a vector from "Vector_Var.txt" and reduces its dimensionality to a given dimension. The resulting vector is returned as a table.
Vector.forwardprop_vectors(input_vector, weight_matrix, bias_vector)
This function performs forward propagation on an input vector using a given weight matrix and bias vector. The resulting output vector is returned as a table.
Vector.activation_func(vector)
This function applies a ReLU activation function to a given vector. The resulting vector is returned as a table.
Vector.backpropagate(output_error, weight_matrix, input_vector)
This function performs backpropagation on a given output error, weight matrix, and input vector. The resulting gradients are returned as a table.
Vector.create_neural_network(num_inputs, num_hidden_layers, hidden_layer_size, num_outputs)
This function creates a neural network with a specified number of inputs, hidden layers, hidden layer size, and outputs. The resulting neural network is returned as a table.
Vector.train_neural_network(neural_network, input_data, output_data, learning_rate, num_epochs)
This function trains a neural network using a specified input dataset, output dataset, learning rate, and number of epochs. The resulting trained neural network is returned as a table.
Vector.evaluate_neural_network(neural_network, input_data, output_data)
This function evaluates the performance of a trained neural network on a specified input dataset and output dataset. The resulting accuracy score is returned as a number.



The Vector Library is a Lua library for managing integer vectors. It provides a set of functions for creating, manipulating, and saving vectors as tables. The library can be used to perform common vector operations such as addition, subtraction, scaling, rotation, normalization, projection, and dimensionality reduction. The library is designed to be flexible and easy to use, with clear documentation and examples provided for each function. It can be used for a wide range of applications, such as machine learning, data analysis, and graphics programming.
 
Vector.check_boolean(boolean_string, vector)

local vector = {1, 2, 3}
local bool_str = "true"

-- Check if the boolean in the boolean string is true for the given vector
local result = Vector.check_boolean(bool_str, vector)

-- If the boolean is true and the vector has a value, returns the vector
-- result = {1, 2, 3}


Vector.create_vector(vector_values)

local vector_values = {1, 2, 3}

-- Create a vector from a list of values
local vector = Vector.create_vector(vector_values)

-- vector = {1, 2, 3}

Vector.get_size(vector)

local vector = {1, 2, 3}

-- Get the size of the vector
local size = Vector.get_size(vector)

-- size = 3



Vector.save_vector(vector)

local vector = {1, 2, 3}

-- Save a vector to a file
Vector.save_vector(vector)

-- The vector is saved to the file "Vector_Var.txt"

Vector.add_vectors()

-- Add multiple vectors stored in Vector_Var.txt and return the result as a vector
local result = Vector.add_vectors()

-- result = {2, 4, 6} if Vector_Var.txt contains vectors {1, 2, 3} and {1, 2, 3}Vector.sub_vectors()

-- Subtract multiple vectors stored in Vector_Var.txt and return the result as a vector
local result = Vector.sub_vectors()

-- result = {0, 0, 0} if Vector_Var.txt contains vectors {1, 2, 3} and {1, 2, 3}


Vector.scaling_vectors(scaling_factor)

-- Scale a vector stored in Vector_Var.txt by a given scaling factor
local scaling_factor = 2
local result = Vector.scaling_vectors(scaling_factor)

-- result = {2, 4, 6} if Vector_Var.txt contains vector {1, 2, 3}


Vector.rotate_vector(rotation_angle)

-- Rotate a vector stored in Vector_Var.txt by a given angle
local rotation_angle = 90
local result = Vector.rotate_vector(rotation_angle)

-- result = {-2, 1, 3} if Vector_Var.txt contains vector {1, 2, 3}

Vector.norm_vectors()

-- Normalize a vector stored in Vector_Var.txt
local result = Vector.norm_vectors()

-- result = {0.2673, 0.5345, 0.8018} if Vector_Var.txt contains vector {1, 2, 3}



Vector.project_vectors(target_vector)

-- Project a vector stored in Vector_Var.txt onto a target vector
local target_vector = {1, 0, 0}
local result = Vector.project_vectors(target_vector)

-- result = {1, 0, 0} if Vector_Var.txt contains vector {1, 2, 3}



Vector.dimreduct_vectors(target_dimension) 

-- Example usage
local vector = {1, 2, 3, 4, 5}
local target_dimension = 3
local reduced_vector = Vector.dimreduct_vectors(target_dimension)

-- The original vector is projected onto a 3D subspace and reduced to a 3D vector
-- If the original vector has more than 3 dimensions, the resulting vector will be a projection onto a 3D subspace of the original vector


Vector.forwardprop_vectors(layers, weights, biases) 

-- Example usage
local layers = {3, 4, 2}
local weights = {
    {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
    {{1, 2}, {3, 4}, {5, 6}, {7, 8}}
}
local biases = {
    {1, 2, 3, 4},
    {5, 6}
}
local input = {1, 2, 3}
local output = Vector.forwardprop_vectors(layers, weights, biases, input)

-- The input vector is forward propagated through a neural network with 3 layers
-- The first layer has 3 input neurons and 4 output neurons, the second layer has 4 input neurons and 2 output neurons, and the third layer has 2 output neurons
-- The weights and biases of each layer are specified by the weights and biases tables
-- The resulting output vector is returned

Note: This function requires the Torch7 library to be installed in your Lua environment


Vector.activation_func(activation_type, input)`

-- Example usage local activation_type = "sigmoid" local input = {1, 2, 3, 4} local output = Vector.activation_func(activation_type, input)
-- The input vector is passed through a sigmoid activation function -- The resulting output vector is returned
Note: This function supports the "sigmoid", "tanh", and "ReLU" activation functions

Vector.backpropagate(layers, weights, biases, output, target_output, learning_rate)

-- Example usage
local layers = {3, 4, 2}
local weights = {
    {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
    {{1, 2}, {3, 4}, {5, 6}, {7, 8}}
}
local biases = {
    {1, 2, 3, 4},
    {5, 6}
}
local output = {0.1, 0.9}
local target_output = {1, 0}
local learning_rate = 0.5
local error = Vector.backpropagate(layers, weights, biases, output, target_output, learning_rate)

-- The output vector of a neural network is compared to a target output vector, and an error value is calculated
-- The weights and biases of each layer are specified by the weights and biases tables
-- The learning rate is a parameter that controls the speed of weight updates during backpropagation
-- The resulting error value is returned

Note: This function requires the Torch7 library to be installed in your Lua environment. Also, this function assumes that the last layer of the neural network uses a softmax activation function.
