import kagglehub
import shutil
import os 

def main():
  if not os.path.exists("./train"):
    # Download kaggle dataset
    path = kagglehub.dataset_download("alexanderyyy/mnist-patched-2022")
    shutil.move(path, '.')

    # Move the train folder to the current directory and delete unwanted folder
    source = "./1/mnist_png_patched/train"
    destination = "./train"
    shutil.move(source, destination)

    shutil.rmtree("./1")

if __name__ == '__main__':
  main()