import kagglehub
import shutil
import os, random

TEST_SAMPLES = 10

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

    # Create a test folder
    os.mkdir("./test")
    for i in range(10):
      os.mkdir(f"./test/{i}")
      path = f"./train/{i}/"
      for j in range(TEST_SAMPLES):
        shutil.move(path + str(random.choice(os.listdir(path))), f"./test/{i}")


if __name__ == '__main__':
  main()