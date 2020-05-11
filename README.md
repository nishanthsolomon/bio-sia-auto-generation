# Bio SIA Auto-Generation Task

## Repository for CSE 576 automatic dataset generation assignment.

1. Install the required packages as in requirements.txt
2. Run python main.py --num_sia_generation --cuda_device --manual_input to analyze automatically create the dataset.

    a. num_sia_generation - number of data points to be created.

        python main.py --num_sia_generation 5

    b. cuda_device - GPU device id, if available.

        python main.py --cuda_device 0

    c. manual_input - add this argument if you want to try with a custom sentence_4 

        python main.py --manual_input

The three inputs are optional, with default values num_sia_generation = 20, cuda_device = -1 and manual_input as false
3. The created dataset would be stored in [dataset/sia_dataset.tsv](./dataset/sia_dataset.tsv)

## dataset folder

1. Create a folder [dataset](./dataset).
2. Place the file [sentence_4.tsv](./dataset/sentence_4.tsv) containing all the sentence 4's seperated by newlines in the dataset folder.

## Manual Input

Add a argument --manual_input in running the python script.

    python main.py --manual_input --cuda_device 0

This will ask the user to input the sentence_4 and would return the query, sentence_4, sentence_3, sentence_2, sentence_1 in a JSON.

## Configuration

The configuration is maintained in [sia_auto_generation/conf/sia_configuration.conf](./sia_auto_generation/conf/sia_configuration.conf)