from helpers import *

def main():
  #get user command line arguments
  args = get_args()
  data_set = args.data_set
  network_size = args.network_size

  #get data from the data set the user chose (cifar10, cifar100, fashion_mnist)
  (x_train, y_train), (x_test, y_test) = get_data(data_set)

  #load the models based on network size user picked
  #cnn_model, vit_model = load_models(data_set, network_size)
  cnn_model = load_models(data_set, network_size)
  #test models based on network size user picked
  #test_models(cnn_model, vit_model, data_set, network_size)
  test_models(cnn_model, network_size, x_test, y_test)
if __name__ ==  "__main__":
  main()