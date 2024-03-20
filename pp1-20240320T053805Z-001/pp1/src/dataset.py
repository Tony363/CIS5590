# BEGIN - DO NOT CHANGE THESE IMPORTS/CONSTANTS OR IMPORT ADDITIONAL PACKAGES.
import torch
from torch.utils.data import Dataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
PADDING_VALUE = 0
UNK_VALUE = 1
# END - DO NOT CHANGE THESE IMPORTS/CONSTANTS OR IMPORT ADDITIONAL PACKAGES.


# split_train_val_test
# This method takes a dataframe and splits it into train/val/test splits.
# It uses the props argument to split the dataset appropriately.
#
# args:
# df - the entire dataset DataFrame 
# props - proportions for each split. the last value of the props array 
#         is repetitive, but we've kept it for clarity.
#
# returns: 
# train DataFrame, val DataFrame, test DataFrame
#
def split_train_val_test(df, props=[.8, .1, .1]):
    assert round(sum(props), 2) == 1 and len(props) >= 2
    train_row = int(df.shape[0]*props[0])
    val_row = train_row + int(df.shape[0]*props[1])
    # test_row = val_row + int(df.shape[0]*props[2])

    train_df, test_df, val_df =\
     df.iloc[:train_row,:],\
     df.iloc[train_row:val_row,:],\
     df.iloc[val_row:,:]
    
    ## YOUR CODE STARTS HERE (~3-5 lines of code) ##
    # hint: you can use df.iloc to slice into specific indexes or ranges.

    ## YOUR CODE ENDS HERE ##
    
    return train_df, val_df, test_df

# generate_vocab_map
# This method takes a dataframe and builds a vocabulary to unique number map.
# It uses the cutoff argument to remove rare words occuring <= cutoff times. 
# *NOTE*: "" and "UNK" are reserved tokens in our vocab that will be useful
# later.
# 
# args:
# df - the entire dataset DataFrame 
# cutoff - we exclude words from the vocab that appear less than or
#          eq to cutoff
#
# returns: 
# vocab - dict[str] = int
#         In vocab, each str is a unique token, and each dict[str] is a 
#         unique integer ID. Only elements that appear > cutoff times appear
#         in vocab.
#
# reversed_vocab - dict[int] = str
#                  A reversed version of vocab, which allows us to retrieve 
#                  words given their unique integer ID. This map will 
#                  allow us to "decode" integer sequences we'll encode using
#                  vocab!
# 
def generate_vocab_map(df, cutoff=2):
    import pandas as pd
    vocab = {"": PADDING_VALUE, "UNK": UNK_VALUE}
    reversed_vocab = None
    
    ## YOUR CODE STARTS HERE (~5-15 lines of code) ##
    # hint: start by iterating over df["tokenized"]

    ## YOUR CODE ENDS HERE ##
    tokens = pd.Series(sum(df["tokenized"], []))
    # Get the frequency count of words
    frequency_count = tokens.value_counts()
    frequency_count = frequency_count[frequency_count > cutoff].to_dict()
    for i,tok in enumerate(frequency_count):
        frequency_count[tok] = i + 2

    vocab.update(frequency_count)
    reversed_vocab = {value: key for key, value in vocab.items()}
    return vocab, reversed_vocab

# HeadlineDataset
# This class takes a Pandas DataFrame and wraps in a Torch Dataset.
# Read more about Torch Datasets here: 
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# 
class HeadlineDataset(Dataset):
    """
    class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        features = row[1:]
        label = row[0]
        return features, label

    def __len__(self):
        return len(self.dataframe)
    """
    # initialize this class with appropriate instance variables
    def __init__(self, vocab, df, max_length=50):
        # For this method: I would *strongly* recommend storing the dataframe 
        #                  itself as an instance variable, and keeping this method
        #                  very simple. Leave processing to __getitem__. 
        #              
        #                  Sometimes, however, it does make sense to preprocess in 
        #                  __init__. If you are curious as to why, read the aside at the 
        #                  bottom of this file.
        # 
        
        ## YOUR CODE STARTS HERE (~3 lines of code) ##
        import pandas as pd
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        return 
        ## YOUR CODE ENDS HERE ##
    
    # return the length of the dataframe instance variable
    def __len__(self):
        ## YOUR CODE STARTS HERE (1 line of code) ##
        return  len(self.df)
        ## YOUR CODE ENDS HERE ##
    
    # __getitem__
    # 
    # Converts a dataframe row (row["tokenized"]) to an encoded torch LongTensor,
    # using our vocab map we created using generate_vocab_map. Restricts the encoded 
    # headline length to max_length.
    # 
    # The purpose of this method is to convert the row - a list of words - into
    # a corresponding list of numbers.
    #
    # i.e. using a map of {"hi": 2, "hello": 3, "UNK": 0}
    # this list ["hi", "hello", "NOT_IN_DICT"] will turn into [2, 3, 0]
    #
    # returns: 
    # tokenized_word_tensor - torch.LongTensor 
    #                         A 1D tensor of type Long, that has each
    #                         token in the dataframe mapped to a number.
    #                         These numbers are retrieved from the vocab_map
    #                         we created in generate_vocab_map. 
    # 
    #                         **IMPORTANT**: if we filtered out the word 
    #                         because it's infrequent (and it doesn't exist 
    #                         in the vocab) we need to replace it w/ the UNK 
    #                         token
    # 
    # curr_label            - int
    #                         Binary 0/1 label retrieved from the DataFrame.
    # 
    def __getitem__(self, index: int):
        tokenized_word_tensor = torch.tensor([
            self.vocab[t] if self.vocab.get(t) is not None else 0
            for t in self.df.loc[index,'tokenized']
        ],dtype=torch.long)
        curr_label = self.df.label[index]
        ## YOUR CODE STARTS HERE (~3-7 lines of code) ##
        ## YOUR CODE ENDS HERE ##
        return tokenized_word_tensor, curr_label

# collate_fn
# This function is passed as a parameter to Torch DataSampler. collate_fn collects
# batched rows, in the form of tuples, from a DataLoader and applies some final 
# pre-processing.
#
# Objective:
# In our case, we need to take the batched input array of 1D tokenized_word_tensors, 
# and create a 2D tensor that's padded to be the max length from all our tokenized_word_tensors 
# in a batch. We're moving from a Python array of tuples, to a padded 2D tensor. 
#
# *HINT*: you're allowed to use torch.nn.utils.rnn.pad_sequence (ALREADY IMPORTED)
# 
# Finally, you can read more about collate_fn here: https://pytorch.org/docs/stable/data.html
#
# args: 
# batch - PythonArray[tuple(tokenized_word_tensor: 1D Torch.LongTensor, curr_label: int)]
#         len(batch) == BATCH_SIZE
# 
# returns:
# padded_tokens - 2D LongTensor of shape (BATCH_SIZE, max len of all tokenized_word_tensor))
# y_labels      - 1D FloatTensor of shape (BATCH_SIZE)
# 
def collate_fn(batch, padding_value=PADDING_VALUE):
    ## YOUR CODE STARTS HERE (~4-8 lines of code) ##
    ## YOUR CODE ENDS HERE ##
    data, labels = zip(*batch)  # Unzip the batch
    padded_tokens = pad_sequence(data,padding_value=padding_value,batch_first=True)  # Stack images to create a batch
    labels = torch.tensor(labels)  # Convert labels into a tensor
    return padded_tokens,labels

#
# Completely optional aside on preprocessing in __init__.
# 
# Sometimes the compute bottleneck actually ends up being in __getitem__.
# In this case, you'd loop over your dataset in __init__, passing data 
# to __getitem__ and storing it in another instance variable. Then,
# you can simply return the preprocessed data in __getitem__ instead of
# doing the preprocessing.
# 
# There is a tradeoff though: can you think of one?
# 