# Comment Classifier

![sents](/images/sentiment_analysis.jpg)

<br>


## Technologies Used
* Python
* Pandas
* NumPy
* Scikit-Learn
* NLTK
* SpaCy
* Gensim
* TensorFlow
* AWS
* Jupyter
* Matplotlib
* Seaborn

![logos](/images/logos.png)

<br>

## Summary
#### For this project I used Natural Language Processing to create a tool that classifies text into 1 of 6 emotion classes.  I used Google's GoEmotions dataset which was created using comments from Reddit.com labeled for 27 emotions and Neutral.  The dataset was originally intended for multi-class classification but I used it for single-class classification.  To help account for the class imbalance and the similarity of several classes I started by consolidating the 28 classes into 6.  Next I built a text cleaner which performed as well as other widely used text cleaners.  After that I grid searched several models, tested an LSTM-RNN, and explored feature engineering and error analysis techniques to understand how the model was working to be able to improve it.  Finally, I created a function, a Comment Classifier, that takes in a comment and returns the top three choices for how the comment is classified and how certain the model is about each of those choices.


<br>

## The Data
#### The GoEmotions dataset was created by Google using a collection of comments from Reddit.com which they labeled for 27 emotions and Neutral.  Using Reddit.com meant that many comments were quite unpleasant in one way or another but my model should still work for any text.  I changed the multi-classes to single-classes ("*Joy, Happiness, Neutral*" became just "*Joy*") and consolidated the 28 classes into 6 because so many comments were classified as *Neutral* and many of the classes had overlapping qualities.


![eda](/images/eda.png)
![consolidation](/images/ClassConsolidation.jpeg)


<br>

## Building and Improving the Model
#### To build and improve the model I cycled through cleaning text, testing and tuning models, evaluating results, and starting over making adjustments using what I learned. 

![nlp](/images/nlp_pipeline.png)

#### I created a text cleaner to do the following:

* Spellcheck 
* Lowercase 
* Remove puncuation
* Remove numbers
* Remove new lines
* Remove URLs
* Remove stopwords
* Lemmatize

#### After cleaning the text I'd print a graph of the most common words and their counts in the cleaned corpus and in several classes.  At that point I'd try to assess how useful each word was for the model.  I'd also sometimes print out examples of a word in the context it appears in the corpus to get a better idea of how it might affect analysis.  Then, I'd gridsearch and tune models using some error analysis techniques described below as part of checking the results.  Finally, I'd return to text cleaning and feature engineering and make adjustments such as to my stopwords list or vectorizer.

![graph](/images/common_words_graph.png)
![context](/images/context_examples.png)

<br>

### Error Analysis
#### I found an error analysis technique to be helpful when trying to know more about how the model made its choices.  The technique uses heatmapped confusion matrices with grayscaled and inverted colors.  Below, in the matrix on the left, in the boxes running diagonal from top left to bottom right, the brighter the box is the more comments are being correctly classified for that class.  In the matrix on the right, the brighter a box is the more comments on the y axis there are being misclassified into the class on the x axis.  The matrices show that many comments which should be classified as *Sadness* or *Excitement* are being classified as *Neutral*.  I think this is because a few features of this dataset are significant challenges for the model like the large class imbalance, a lack of data in the *Sadness* and *Excitement* classes, and the labeling issue which I found summarized well in [this blog post](https://koaning.io/posts/labels/). 

![error_analysis](/images/error_analysis_slide.jpeg)

<br>

## Final Model
#### The XGBClassifier model performed best reaching 54% accuracy.  Much better than random guessing with 6 classes but it still leaves room for improvement.  

<br>


## Next Steps
#### Next steps for this project are to take on the classification challenge with transformer models like XLNet, GPT-3, and BERT.

<br>
