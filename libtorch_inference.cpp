#include </home/mlda/shobhit/pytorch/torch/script.h> // One-stop header.

#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, const char* argv[]) {
  if (argc != 4) 
  {
    std::cerr << "usage: inferencer <path-to-exported-script-module>\n";
    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

  // Read image and convert it to RGB and resize
  Mat image_bgr, image;
  image_bgr = imread(argv[2]);
  cvtColor(image_bgr, image, COLOR_BGR2RGB);
  resize(image, image, Size(1920, 1080));

  /*for (int j=0;j<10;j++)
  {
    cout<<image.at<Vec3b>(0,j)<<endl;
  }*/

  // The channel dimension is the last dimension in OpenCV
  at::Tensor tensor_image = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, at::kByte);
  tensor_image = tensor_image.to(at::kFloat);

  // Transpose the image for [channels, rows, columns] format of pytorch tensor
  tensor_image = at::transpose(tensor_image, 1, 2);
  tensor_image = at::transpose(tensor_image, 1, 3);
  cout<<tensor_image.size(1)<<" "<<tensor_image.size(2)<<" "<<tensor_image.size(3)<<endl; //3,1080,1920
  

  // Create a vector of torch inputs
  std::vector<torch::jit::IValue> input;
  input.emplace_back(tensor_image);

  // Execute the model and turn its output into a tensor.
  auto output = module->forward(input).toTensor().clone().squeeze(0);
  output = output.to(at::kByte); // Convert the values to byte [0-255]
  
  // OpenCV cannot build a Mat directly from 3 channels so make a separate Mat for each channel and then merge
  auto red_output = output.slice(0, 0, 1);
  auto green_output = output.slice(0, 1, 2);
  auto blue_output = output.slice(0, 2, 3);
  
  // Initialize OpenCV variables
  Mat red_output_mat({1920, 1080}, CV_8UC1, red_output.data<uint8_t>());
  Mat green_output_mat({1920, 1080}, CV_8UC1, green_output.data<uint8_t>());
  Mat blue_output_mat({1920, 1080}, CV_8UC1, blue_output.data<uint8_t>());
  
  Mat final_img;
  vector<Mat> channels;
  channels.push_back(red_output_mat);
  channels.push_back(green_output_mat);
  channels.push_back(blue_output_mat);
  cv::merge(channels, final_img);
  cout<<fin_img.rows<<" "<<fin_img.cols<<endl;
  
  Mat fin_img_bgr;
  cvtColor(fin_img, fin_img_bgr, COLOR_RGB2BGR);
  imwrite(argv[3], fin_img_bgr);
  //std::cout << output.slice(/*dim=*/3, /*start=*/0, /*end=*/2) << '\n';

