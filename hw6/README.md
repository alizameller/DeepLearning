I used the tutorial from this website: https://machinelearningmastery.com/implementing-the-transformer-encoder-from-scratch-in-tensorflow-and-keras/ to help 
guide my implementation and also referenced the paper "Attention Is All You Need" 
as recommended in the assignment description. I also worked with my classmate, 
David Stekol, on parts of the assignment. 

To demonstrate that my transformer works, I showed that by overfitting to a 
specific sentence, the transformer can predict the next word in the sentence
during testing. In my test, I showed that by training on "hi prof curro this 
is my transformer", the transformer was able to predict each successive token 
when testing with "hi prof curro." To accomplish this, I created a loop that 
fed back the predicted next word into the input in order to achieve successful
prediction of the words remaining in the sentence. This means that each word
predicted by the model was successful regardless of which word the model was
predicting (i.e. where in the sentence it had to predict). However, because 
this is a decoder only model, the transformer can only predict future tokens.

This brings us to masked multi-head attention. I tested the masking using a 
derivative test which checked that a predicted token in the output and a 
token that appeared later on in the input sentence were independent. By taking
the jacobian of the output matrix, the matrices corresponding to the tokens
each had rows of zeroes occupying the row representing the tokens it did not
depend on (the last token did not have any rows of zeroes). The fact that this
test passes demonstrates the causal mask works. 
