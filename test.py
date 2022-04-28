from helpers import *

def main():
  #get user command line arguments
  args = get_args()
  data_set = args.data_set
  network_size = args.network_size

  #get data from the data set the user chose (cifar10, cifar100, fashion_mnist)
  (x_train, y_train), (x_test, y_test) = get_data(data_set)
  input_shape = x_train[0].shape
  
  #create the CNN and vision transformer based on network size user picked
  cnn_arch, vit_arch = create_models(data_set, network_size, input_shape, x_train)
  #load the models based on network size user picked
  cnn_model, vit_model = load_models(cnn_arch, vit_arch, data_set, network_size)
  #test models based on network size user picked
  test_models(cnn_model, vit_model, x_test, y_test)

if __name__ ==  "__main__":
  main()