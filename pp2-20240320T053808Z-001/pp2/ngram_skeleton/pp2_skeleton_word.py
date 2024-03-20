import math, random
from typing import List, Tuple

################################################################################
# Part 0: Utility Functions
################################################################################

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return ['~'] * n

Pair = Tuple[str, str]
Ngrams = List[Pair]
def ngrams(n, text:str) -> Ngrams:
    text=text.strip().split()
    ''' Returns the ngrams of the text as tuples where the first element is
        the n-word sequence (i.e. "I love machine") context and the second is the word '''
    return [
        (start_pad(n) + text[i:i+n] if n != 0 else 0, text[i]) 
        for i in range(len(text))
    ]

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8') as f:
        model.update(f.read())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################
"""
class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.vocab = set()
        self.count = {}

    def get_vocab(self):
        ''' Returns the set of words in the vocab '''
        pass

    def update(self, text:str):
        ''' Updates the model n-grams based on text '''
        pass

    def prob(self, context:str, word:str):
        ''' Returns the probability of word appearing after context '''
        pass

    def random_word(self, context):
        ''' Returns a random word based on the given context and the
            n-grams learned by this model '''
#         random.seed(1)
        pass

    def random_text(self, length):
        ''' Returns text of the specified word length based on the
            n-grams learned by this model '''
        pass

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        pass
"""

class NgramModel(object):
    def __init__(self, n, k=1):
        self.n = n
        self.k = k  # Smoothing parameter
        self.ngrams = {}  # To store n-gram counts
        self.vocab = set()  # To store the vocabulary (unique characters)

    def get_vocab(self):
        return self.vocab

    def update(self, text):
        # Update vocabulary
        text = text.strip().split()
        self.vocab.update(set(text))
        
        # Update n-gram counts
        for i in range(len(text) - self.n + 1):
            # Extract the n-gram and context
            ngram = text[i:i+self.n]
            context = ' '.join(ngram[:-1])
            char = ngram[-1]
            
            if context not in self.ngrams:
                self.ngrams[context] = {}
            
            if char not in self.ngrams[context]:
                self.ngrams[context][char] = 0
            
            self.ngrams[context][char] += 1

    def prob(self, context, char):
        # If the context hasn't been seen before, return uniform probability
        if context not in self.ngrams or len(self.ngrams[context]) == 0:
            return 1 / len(self.vocab)
        
        # Calculate probability with Laplace smoothing
        char_count = self.ngrams[context].get(char, 0)
        total_count = sum(self.ngrams[context].values())
        return (char_count + self.k) / (total_count + self.k * len(self.vocab))

    def random_word(self, context):
        # Generate a random number r
        r = random.random()
        
        # Sort the vocabulary
        sorted_vocab = sorted(list(self.vocab))
        
        # Calculate the cumulative probability until it exceeds r
        cumulative_prob = 0
        for char in sorted_vocab:
            cumulative_prob += self.prob(context, char)
            if cumulative_prob > r:
                return char
        return sorted_vocab[-1] 
    
    def random_text(self, length):
        if self.n == 0:  # If n=0, context is always empty
            context = ''
        else:
            context = ' ' * (self.n - 1)  # Starting context is n-1 spaces
        
        generated_text = ''
        for _ in range(length):
            next_char = self.random_word(context)
            generated_text += next_char + ' '
            
            if self.n > 0:  # Update context if n > 0
                context = context[1:] + next_char  # Slide the context window
            # For n=0, context remains an empty string
            
        return generated_text

    def perplexity(self, text):
        padded_text = ' ' * (self.n - 1) + text  # Add padding for the start of the text
        log_probability_sum = 0
        N = len(text)
        
        for i in range(N):
            context = padded_text[i:i+self.n-1]
            char = padded_text[i+self.n-1]
            probability = self.prob(context, char)
            
            # Check for zero probability to avoid math domain error
            if probability == 0:
                return float('inf')
            
            log_probability_sum += math.log(probability)
        
        # Calculate perplexity using the log probability
        perplexity = math.exp(-log_probability_sum / N)
        return perplexity




################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################
"""
class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        pass

    def get_vocab(self):
        pass

    def update(self, text:str):
        pass

    def prob(self, context:str, word:str):
        pass
"""
class NgramModelWithInterpolation(NgramModel):
    def __init__(self, n, k=1):
        super().__init__(n, k)
        # Initialize lambda values for interpolation with equal weights by default
        self.lambdas = [1.0 / (n + 1)] * (n + 1)
    
    def set_lambdas(self, lambdas):
        if len(lambdas) != self.n + 1:
            raise ValueError("Number of lambda values must be equal to n + 1.")
        if abs(sum(lambdas) - 1.0) > 1e-6:
            raise ValueError("Sum of lambda values must be 1.")
        self.lambdas = lambdas
    
    def prob(self, context, char):
        # Implement the interpolated probability calculation
        interpolated_prob = 0
        for i in range(len(self.lambdas)):
            unigram_context = '' if i == 0 else context[-i:]
            
            # Calculate the probability with smoothing for the given context length
            char_count = self.ngrams.get(unigram_context, {}).get(char, 0)
            context_count = sum(self.ngrams.get(unigram_context, {}).values())
            lambda_i = self.lambdas[i]
            vocab_size = len(self.vocab)
            smoothed_prob = (char_count + self.k) / (context_count + self.k * vocab_size)
            
            # Weight the probability by the corresponding lambda value
            interpolated_prob += lambda_i * smoothed_prob
        
        return interpolated_prob
################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass