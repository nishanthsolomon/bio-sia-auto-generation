# Bio SIA Auto-Generation Task

## Repository for CSE 576 automatic dataset generation assignment.

1. Install the required packages as in requirements.txt
2. Run python main.py --num_sia_generation --cuda_device to analyze automatically create the dataset.\
    a. num_sia_generation - number of data points to be created.\
    b. cuda_device - GPU device id, if available.\
The two inputs are optional, with default values num_sia_generation = 20 and cuda_device = -1
3. The created dataset would be stored in [dataset/sia_dataset.tsv](./dataset/sia_dataset.tsv)

## dataset folder

1. Create a folder [dataset](./dataset).
2. Place the file [sentence_4.tsv](./dataset/sentence_4.tsv) containing all the sentence 4's seperated by newlines in the dataset folder.