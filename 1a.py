# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:40:49 2020

@author: Giannis
"""

import pandas as pd
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

df = pd.read_csv("train.csv");
text = df.Content[df['Label'] == 'Business']
text2 = df.Content[df['Label'] == 'Entertainment']
text3 = df.Content[df['Label'] == 'Health']
text4 = df.Content[df['Label'] == 'Technology']
wordcloud = WordCloud(width = 1200, height = 1000,max_font_size=200, max_words=300, stopwords = STOPWORDS, background_color="white").generate(str(text))
wordcloud2 = WordCloud(width = 1200, height = 1000,max_font_size=200, max_words=300, stopwords = STOPWORDS, background_color="white").generate(str(text2))
wordcloud3 = WordCloud(width = 1200, height = 1000,max_font_size=200, max_words=300, stopwords = STOPWORDS, background_color="white").generate(str(text3))
wordcloud4 = WordCloud(width = 1200, height = 1000,max_font_size=200, max_words=300, stopwords = STOPWORDS, background_color="white").generate(str(text4))

plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(wordcloud2, interpolation="bilinear")
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(wordcloud3, interpolation="bilinear")
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(wordcloud4, interpolation="bilinear")
plt.axis("off")
plt.show()


"""

df = pd.read_csv(r"train.csv", encoding ="latin-1") 
  
comment_words = ' '
stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
for val in df.Content: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
          
    for words in tokens: 
        comment_words = comment_words + words + ' '
  
  
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
"""