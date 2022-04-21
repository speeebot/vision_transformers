from helpers import *

def main():
  args = get_args()

  data_set = args.data_set
  network_size = args.network_size

  (x_train, y_train), (x_test, y_test) = get_data(data_set)
  input_shape = x_train[0].shape

  if data_set == "cifar10" or data_set == "fashion_mnist":
    num_classes = 10
  elif data_set == "cifar100":
    num_classes = 100

  if network_size == "tiny":
    train_tiny(num_classes, input_shape, x_train, y_train, x_test, y_test)
  elif network_size == "small":
    train_small(num_classes, input_shape, x_train, y_train, x_test, y_test)
  elif network_size == "base":
    train_base(num_classes, input_shape, x_train, y_train, x_test, y_test)

  print(f"data_set: {data_set} input_shape: {input_shape}")
  print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
  print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

if __name__ ==  "__main__":
  main()