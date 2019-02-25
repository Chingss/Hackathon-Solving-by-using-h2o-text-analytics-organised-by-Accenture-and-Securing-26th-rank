# Hackathon-Solving-by-using-h2o-text-analytics-organised-by-Accenture-and-Securing-26th-rank
## Solved with R programming  Solving methodology for data wrangling involves following steps :  
1. Started Impoting the data in RStudio. 
2. Then looked the class and structure of the both train and text dataset features. 
3. Converted the require class of features. 
4. After looking into problem statement found that there is need of converting the text variable into some numeric values 
   for the score prediction purpose. 
   So for that i first tried with tidytext package of R but it was not successfull to explain the text variable.     
   After that i reasearche about the availablity of algorithms that can be used for this kind of problem and finally 
   found out a package which is correctly explaing the variance in sentiments.  
5. Explaination about the package      
To demonstrate how sentiment analysis works, we’ll use the [SentimentAnalysis()](https://cran.r-project.org/web/packages/SentimentAnalysis/index.html) package in R. 
This implementation utilizes various existing dictionaries, such as Harvard IV, QDAP, Loughran-McDonald, and DictionaryHE, 
which is a “dictionary with opinionated words from Henry’s Financial dictionary.” In addition, you can create customized dictionaries. 
In our example, we’ll use the acq data set from the tm package. This package holds 50 news articles from the Reuters-21578 data set. 
All documents belong to the topic of dealing with corporate acquisitions.


After Getting the score of the comment variables
Now the next task was to summarize the data and by summarizing i found out that it contains missing values in both Train and text data 
to resolve that i implemented the [missForest()](https://www.rdocumentation.org/packages/missForest/versions/1.4) 



