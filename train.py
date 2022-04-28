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
  cnn_model, vit_model = create_models(data_set, network_size, input_shape, x_train)
  #train CNN and Vision Transformer based on network size user picked
  cnn_history, vit_history = train_models(cnn_model, vit_model, x_train, y_train)
  #save the models
  save_models(cnn_model, vit_model, data_set, network_size)
  #save learning curves
  plot_results(cnn_history, data_set, network_size, "CNN")
  plot_results(vit_history, data_set, network_size, "ViT")

if __name__ ==  "__main__":
  main()