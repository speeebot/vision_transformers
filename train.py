from helpers import *

def main():
  train_or_test = "train"
  args = get_args()

  data_set = args.data_set
  network_size = args.network_size
  num_classes = 10

  (x_train, y_train), (x_test, y_test) = get_data(data_set)
  input_shape = x_train[0].shape

  if network_size == "tiny":
    run_tiny(train_or_test, num_classes, input_shape, x_train, y_train, x_test, y_test)
  elif network_size == "small":
    run_small()
  elif network_size == "base":
    run_base()

  print(f"data_set: {data_set} input_shape: {input_shape}")
  print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
  print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

if __name__ ==  "__main__":
  main()