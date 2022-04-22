from helpers import *

def main():
  #get user command line arguments
  args = get_args()
  data_set = args.data_set
  network_size = args.network_size

  print("Loading Test Data")
  (x_train, y_train), (x_test, y_test) = get_data(data_set)
  input_shape = x_train[0].shape

  if network_size == "tiny":
    test_tiny(x_test, y_test)
  elif network_size == "small":
    test_small(x_test, y_test)
  elif network_size == "base":
    test_base(x_test, y_test)

  print(f"data_set: {data_set} input_shape: {input_shape}")
  print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
  print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

if __name__ ==  "__main__":
  main()