# 
# title: "Capstone Project - Data Science Professional Certificate"
# author: "Werner Alencar Advincula Dassuncao"


# README Info on project, objetives and motivation
# 
# Losses due to fraudulent payments have reached globally \$ 28.65 billion in 2019, according to the most recent *Nilson Report* data. The United States alone accounts for over a third of the worldwide loss. These numbers are quite high and estimates for the US in 2020 are somewhere around \$ 11 billion due to credit card fraud says *Julie Conroy*, research director for Aite Group's fraud and anti-money laundering practice. These fraud cases affect consumers, merchants and card issuers alike. The total of cost for credit card fraud extends far beyond the cost of the illegally purchased goods. So, being able to detect fraud before it happens is extremely important.
# 
# Access to actual financial data for research outside corporations is blocked due to privacy. For this project we will work with a data set from *The Mobile Money Payment Simulation* which was a case study based on a real company that has developed a mobile money implementation that provides mobile phone users with the ability to transfer money between themselves using the phone as a sort of electronic wallet.  Edgar Alonso Lopez-Roza explains:  *"The development of PaySim covers two phases. During the first phase, we modeled and implemented a MABS (Multi Agent Based Simulation) that used the schema of the real mobile money service and generated synthetic data following scenarios that were based on predictions of what could be possible when the real system starts operating. During the second phase we got access to transactional financial logs of the system and developed a new version of the simulator which uses aggregated transactional data to generate financial information more alike the original source"*. 
# 
# This project aims to generate a machine learning model to predict if a transaction is fraudulent. Before we can talk about machine learning we need to explore, summarize and graph the data with the objective of learning about possible patterns and/or correlation between the variables. Secondly, we will look into the available predicting candidates (features).  For this project our target variable will point if a observation is fraud or not.
# 
# Next will dive into two models for the machine learning section. The first will be Support Vector Machines (SVM) where we will deploy linear and polynomial kernels, and, the later, eXtreme Gradient Boosting (xgboost). These models will be trained using our train set, tested with the test set, and finally their performance will be "double-checked" on our final hold-out validation set.


## ----install packages, include = FALSE, echo = FALSE, warning=FALSE, message=FALSE------------------
if(!require(tidyverse)) install.packages('tidyverse', repos = 'http://cran.us.r-project.org')
if(!require(kableExtra)) install.packages('kableExtra', repos = 'http://cran.us.r-project.org')
if(!require(gridExtra)) install.packages('gridExtra', repos = 'http://cran.us.r-project.org')
if(!require(scales)) install.packages('scales', repos = 'http://cran.us.r-project.org')

if(!require(caret)) install.packages('caret', repos = 'http://cran.us.r-project.org')
if(!require(xgboost)) install.packages('xgboost', repos = 'http://cran.us.r-project.org')
if(!require(Matrix)) install.packages('Matrix', repos = 'http://cran.us.r-project.org')
if(!require(e1071)) install.packages('e1071', repos = 'http://cran.us.r-project.org')
if(!require(DiagrammeR)) install.packages('DiagrammeR', repos = 'http://cran.us.r-project.org')
if(!require(clue)) install.packages('clue', repos = 'http://cran.us.r-project.org')
if(!require(devtools)) install.packages('devtools', repos = 'http://cran.us.r-project.org')
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("ComplexHeatmap")


## ---- warning=FALSE, message=FALSE------------------------------------------------------------------
library(devtools)
library(tidyverse)
library(kableExtra)
library(gridExtra)
library(scales)
library(caret)
library(xgboost)
library(Matrix)
library(e1071)
library(clue)
library(DiagrammeR)
library(ComplexHeatmap)


################################################################################
# NOTE REGARDING TEMPORARY LINKS FROM KAGGLE.COM
# 
# The PaySim dataset is publicly hosted by Kaggle through the URL: *https://www.kaggle.com/ntnu-testimon/paysim1/download*. 
# I wrote a R script to automatically download the data, decompress the zip file and load it up to the R environment, 
# which worked while the link did not expire. This code is available bellow in the code chunk *download the source data*. 
# Kaggle does not provide a direct download link for this dataset, therefore the automated process to download, 
# extract and load its data is not possible as per April 16th, 2021.  This is due to the fact that the download link 
# is dynamically created to the user connection (session) and expires shortly after clicking the download button. 
# 
## ----download the source data, warning = FALSE, message=FALSE---------------------------------------
# # KAGGLE's DOWNLOAD LINK EXPIRES SHORTLY AFTER CLICKING THE DOWNLOAD BUTTON.
# # This is an example code for automating the download, decompressing 
# # and loading of data when a permanent download link is available

# # Create a temporary file
# dl <- tempfile()

# # Paste the direct download link for your connection on Kaggle
# file_direct_link <- 'https://storage.googleapis.com/kaggle-data-sets/1069/1940/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210325%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210325T223835Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=0b6e33ce1e2defe58609a20ff72eb3d7fd227abd4dc9de6a70d6eb133aee721a45f1d6af416c726f9cc4477371ac64d669b28357c88b13fc330c64c1db58898e26d7616e90a4e7a170426c239756256f9fa1c61a91ef931d84d1a4ff1fb9a5b3e65853c8475cfaa5b5ddbe361b873f0aab2b1421f677f27a0e13cfeecf7731a2c40490446341bfcf326537b0e739a26e28fa3b03bd44e5ff027610b45630dfe1b85a2330cbc56e88707be1d22f106e267289ee99893b15e2e8ae6b8ee9a6776c98f1f26ee1da253f2b0338ada70867386b503994d2ab763598388ecc911b64e4e8cd8fa835d4f46a2f7855ea2247e70fb8167a0d1404c7b06db1a1895322e207'

# # Download the file to the temporary file
# download.file( file_direct_link, dl )

# # Decompress the downloaded zip file
# data <- unzip( zipfile = dl, 'PS_20174392719_1491204439457_log.csv' )

# # Read the csv file
# data <- read.csv(data)

# # remove temp file
# unlink(dl)

# Now the "data" object contains the contents from the file downloaded from Kaggle!
################################################################################


# link to Kaggle site: https://www.kaggle.com/ntnu-testimon/paysim1/download
# download the data and extract the zip file to the same folder as the Rmd and R scripts to run this code
## ----load the data----------------------------------------------------------------------------------
# Load the data directly from local csv file
data <- read.csv('PS_20174392719_1491204439457_log.csv')


## ----show data structure, echo= FALSE---------------------------------------------------------------
str(data)

## ----NA_check---------------------------------------------------------------------------------------
anyNA(data)

## ----sample_row, echo=FALSE, warning=FALSE, message=FALSE-------------------------------------------
slice_sample(data) # %>% kable()


## ----data summary, echo=FALSE, warning=FALSE, message=FALSE-----------------------------------------
summary(data) 


## ----create color scale, echo=FALSE, warning=FALSE, message=FALSE-----------------------------------
# We will create a custom color scale so that each transaction type always receives the same color for ease of comparison of plots by quantity and amount of transactions.

# Change the datatype for the type column
data$type <- as.factor(data$type)

library(RColorBrewer)
myColors <- brewer.pal(5,'Set1')
names(myColors) <- levels(data$type)
colScale <- scale_colour_manual(name = 'type', values = myColors)



####### DATA EXPLORATION AND VISUALIZATION ##########


## ----plot 1-3, echo=FALSE, warning=FALSE, message=FALSE---------------------------------------------
# Plot of type by frequency
p1 <- data %>% 
  group_by(type) %>%
  summarize(n = n()) %>%
  mutate(perc = n / sum(n),
         type = reorder(type, perc)) %>%
  ggplot(aes(x = type, y = perc, fill = myColors)) +
  geom_bar(stat = 'identity', show.legend = FALSE)+
  geom_text(aes(label = paste0(round(perc*100, 2), ' %'),
              y = perc), vjust = -.25) +
  labs(title = 'Frequency of transactions by type', x = '', y = '') +
  scale_y_continuous(labels = scales::percent, limits = c(0, .37))

# Plot of type by average amount
p2 <- data %>% 
  group_by(type) %>%
  summarize(average = mean(amount)) %>%
  mutate(average = average,
         type = reorder(type, average)) %>%
  ggplot(aes(x = type, y = average,  fill = myColors)) +
  geom_bar(stat = 'identity', show.legend = FALSE)+
  geom_text(aes(label = paste0('$ ',format(round(average), big.mark = ',', scientific = FALSE)),
              y = average), vjust = -.25) +
  labs(title = 'Transactions per average amount', x = '', y = '') +
  scale_y_continuous(labels = scales::dollar, limits = c(0, 1e6))

# custom global setting 
options(stringsAsFactors = FALSE)

# Boxplot of types and distribution of amounts
p3 <- data %>% mutate( type = factor(type, levels = c('DEBIT','PAYMENT','CASH_IN','CASH_OUT','TRANSFER'))) %>%
  ggplot(aes(x = type, y = amount, color = type)) +
  geom_boxplot(alpha = 0.01, show.legend = FALSE) +
  labs(title = 'Boxplot of transactions by type', x = '', y = 'log10 (amount)') +
  scale_y_continuous(trans = 'log10')  +
  #theme(axis.text.x = element_text(angle = 45, vjust = 0.5)) + 
  scale_color_manual(breaks = c('DEBIT','PAYMENT','CASH_IN','CASH_OUT','TRANSFER'),
                     values = c("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00"))


## ----types plots 1 & 2, echo = FALSE, warning=FALSE, message=FALSE, fig.cap='Distributions by transactional frequency and average amount', fig.width=12----
# Arrange plots 1 and 2 side-by-side
grid.arrange(p1, p2, nrow = 1)


## ----types plot 3, echo = FALSE, warning=FALSE, message=FALSE, fig.cap='Box plot of transactions by type', fig.width=12----
# display plot 3
p3


## ----plots 4-7, echo=FALSE, warning=FALSE, message=FALSE--------------------------------------------
p4 <- data %>% 
  mutate(isFraud = factor(isFraud, levels = c(0,1), labels = c('Legitimate', 'Fraud'))) %>%
  group_by(isFraud) %>%
  summarize(n = n()) %>%
  mutate(perc = n / sum(n),
         isFraud = reorder(isFraud, perc)) %>%
  ggplot(aes(x = isFraud, y = perc, fill = isFraud)) +
  geom_bar(stat = 'identity', show.legend = FALSE)+
  geom_text(aes(label = paste0(round(perc*100, 2), '%'),
              y = perc), vjust = -.25) +
  labs(title = 'Fraud percentage', x = '', y = '') +
  scale_y_continuous(labels = scales::percent) 

p5 <- data %>% filter(isFraud == 1) %>%
  group_by(type) %>%
  summarize(average = mean(amount)) %>%
  mutate(average = average,
         type = reorder(type, average)) %>%
  ggplot(aes(x = type, y = average, fill = factor(type))) +
  geom_bar(stat = 'identity', show.legend = FALSE)+
  geom_text(aes(label = paste0('$ ',format(round(average), big.mark = ',', scientific = FALSE)),
              y = average), vjust = -.25) +
  labs(title = 'Fraud by average amount', x = '', y = '') +
  scale_y_continuous(labels = unit_format(unit = "M", scale = 1e-6)) 

p6 <- data %>% filter(isFraud == 1) %>%
  group_by(type) %>%
  summarize(n = n()) %>%
  mutate(perc = n / sum(n),
         type = reorder(type, perc)) %>%
  ggplot(aes(x = type, y = perc, fill = type)) +
  geom_bar(stat = 'identity', show.legend = FALSE) +
  geom_text(aes(label = paste0(round(perc*100, 2), '%'),
              y = perc), vjust = -.25) +
  labs(title = 'Fraud per type', x = '', y = '') +
  scale_y_continuous(labels = scales::percent) 

p7 <- data %>% filter(isFraud == 1) %>%
  ggplot(aes(x = factor(type), y = amount, colour = type)) +
  geom_boxplot(alpha = 0.1, show.legend = FALSE) +
  labs(title = 'Fraud boxplot', x = '', y = '') +
  scale_y_continuous(labels = unit_format(unit = 'M', scale = 1e-6))


## ----fraud_plots, echo=FALSE,warning=FALSE, message=FALSE, fig.width= 12----------------------------
# Arrange plots 4 and 5
grid.arrange(p4, p5, nrow = 1)


## ----fraud_plots2, echo=FALSE,warning=FALSE, message=FALSE, fig.width= 12---------------------------
# Arrange plots 6 and 7
grid.arrange(p6, p7, nrow = 1)


## ----house_cleaning, echo = FALSE, message = FALSE, warning = FALSE---------------------------------
# remove no longer needed objects to free up memory
rm(p1,p2,p3,p4,p5,p6,p7)


## ----fraud_stats, echo=FALSE, warning=FALSE, message=FALSE------------------------------------------
# create custom overview for fraud transactions
fraud <- data %>% 
  filter(isFraud==1) %>% group_by(type) %>%
  summarize(type = type,
            number = format(n(), big.mark = ',', scientific = FALSE),
            average  = format(mean(amount), big.mark = ',', scientific = FALSE),
            min = format(min(amount), big.mark = ',', scientific = FALSE),
            max = format(max(amount), big.mark = ',', scientific = FALSE),
            sum = format(sum(amount), big.mark = ',', scientific = FALSE) ) %>% 
  distinct(type, number, average, min, max, sum)
fraud %>% kable(caption = 'Summary table of fraudulent transactions')


## ----custom summary fraud_totals, echo = FALSE, warning = FALSE-------------------------------------
# save the number of observations 
total_observations <- nrow(data)

# create a custom summary for fraud transactions
fraud_totals <- data %>% 
  filter(isFraud==1) %>% 
  summarize(n = format(n(), big.mark = ',', scientific = FALSE),
            avg  = format(mean(amount), big.mark = ',', scientific = FALSE),
            min = format(min(amount), big.mark = ',', scientific = FALSE),
            max = format(max(amount), big.mark = ',', scientific = FALSE),
            sum = format(sum(amount), big.mark = ',', scientific = FALSE) ) %>% 
  distinct(n, avg, min, max, sum)


## ---- echo = FALSE----------------------------------------------------------------------------------
# How many observations have the isFraud and isFlaggedFraud set simmultaneously?
data %>% filter ( ( isFraud == 1 ) & ( isFlaggedFraud == 1 ) ) %>% count() %>% pull(n)


## ----echo = FALSE, warning = FALSE, message = FALSE-------------------------------------------------
flagged_totals <- data %>% 
  filter( isFlaggedFraud == 1 ) %>% 
  summarize( quantity = format(n(), big.mark = ',', scientific = FALSE),
            avg  = format(mean(amount), big.mark = ',', scientific = FALSE),
            min = format(min(amount), big.mark = ',', scientific = FALSE),
            max = format(max(amount), big.mark = ',', scientific = FALSE) ) %>% distinct(quantity, avg, min, max)

flagged_totals %>% kable(caption = 'Summary of transactions when isFlaggedFraud is set')


## ----isFlaggedFraud checks , echo = FALSE, warning = FALSE, message = FALSE-------------------------
flagged <- data %>% filter( isFlaggedFraud == 1 )
not_flagged <- data %>% filter( isFlaggedFraud == 0)


## ----questions flagged, include = FALSE, echo = FALSE, warning = FALSE, message = FALSE-------------
print('Are there multiple transactions by the same origin flagged as fraud?')
subset(flagged, nameOrig %in% paste(not_flagged$nameOrig, not_flagged$nameDest))

print('Are there transactions initiated by a destination account flagged as fraud?')
subset(flagged, nameDest %in% not_flagged$nameOrig)

print('Are there multiple transactions where the destination account was flagged?')
subset(flagged, nameDest %in% not_flagged$nameDest)


## ---- echo = FALSE, warning = FALSE, message = FALSE------------------------------------------------
steps <- flagged %>% pull(step)


## ----flagged transactions, echo = FALSE, warning = FALSE--------------------------------------------
print('All cases where the isFlaggedFraud is set')
flagged %>% filter(oldbalanceDest == newbalanceDest,
                   oldbalanceOrg == newbalanceOrig)  %>% 
  kable(caption = 'All cases where the isFlaggedFraud is set')


## ----removing unused objects, echo = FALSE, warning = FALSE-----------------------------------------
# removing no-longer needed objects from memory
rm(flagged, not_flagged)


## ----rename column----------------------------------------------------------------------------------
# For readability and consistency, we will correct the 
# column name "oldbalanceOrg" to "oldbalanceOrig"
data <- data %>% rename(oldbalanceOrig = oldbalanceOrg)


## ----create X and Y objects-------------------------------------------------------------------------
# create a X data object with the actual types which are possible fraud cases
# filter the original data to transfer and cash_out only
X <- data %>% filter( (type == 'TRANSFER') | (type == 'CASH_OUT') )

# grab the target variable 'isFraud' from the filtered dataframe X 
Y <- X$isFraud
# remove 'isFraud' from X
X <- X[,!(names(X) %in% 'isFraud')]

# remove the columns that are not relevant according to data exploration 
drop <- c('nameOrig', 'nameDest', 'isFlaggedFraud')
X <- X[,!(names(X) %in% drop)]

# Encode binary values for TRANSFER and CASH_OUT
X <- X %>% mutate(type = str_replace_all(type, c('TRANSFER' = '0', 'CASH_OUT' = '1')))

# Convert the type to integer
X$type <- as.integer(X$type)

# Remove from memory
rm(data)

# Create two new features(columns) to help train the models
X <- X %>% mutate(balance_error_orig = newbalanceOrig + amount - oldbalanceOrig,
                  balance_error_dest = oldbalanceDest + amount - newbalanceDest)


## ----zero balance percentages destination accounts, echo = FALSE, warning = FALSE-------------------
# get index of all FRAUD transactions
fraud_index <- Y == 1

# create a fraud dataframe
Xfraud <- X[ fraud_index, ]
# create a non-fraud dataframe (note the '!' sign to negate the logical vector)
Xnonfraud <- X[ !fraud_index, ]

# percentage of legitimate zero balance transactions
zero_bal_perc_nonfraud <- Xnonfraud %>% 
  filter( (oldbalanceDest == 0) & (newbalanceDest == 0) & (amount != 0) )  %>%
  summarize( percentage = 100 * n() / nrow(Xnonfraud) ) %>% round(., digits = 2)

# percentage of fraudulent zero balance transactions 
zero_bal_perc_fraud <- Xfraud %>% 
  filter( (oldbalanceDest == 0) & (newbalanceDest == 0) & (amount != 0) )  %>%
  summarize( percentage = 100 * n() / nrow(Xfraud) ) %>% round(., digits = 2) %>%
  .$percentage


## ----highlighting fraud possibilities, echo = FALSE, messate = FALSE, warning = FALSE---------------
# replacing 0 with -1 to highlight the possibility of fraud for the ML.
# filter the dataframe, then select the columns to update, lastly define the desired value 
X[ which( X$oldbalanceDest == 0 & X$newbalanceDest == 0 & X$amount != 0), 
        names(X) %in% c("oldbalanceDest", "newbalanceDest")] <- -1


## ----zero balance percentage origin accounts, echo = FALSE, messate = FALSE, warning = FALSE--------
zero_bal_perc_nonfraud_orig <- Xnonfraud %>% 
  filter( (oldbalanceOrig == 0) & (newbalanceOrig == 0) & (amount != 0) )  %>%
  summarize( percentage = 100 * n() / nrow(Xnonfraud) ) %>% round(., digits = 2)

# percentage of zero balance transactions in case of FRAUD
zero_bal_perc_fraud_orig <- Xfraud %>% 
  filter( (oldbalanceOrig == 0) & (newbalanceOrig == 0) & (amount != 0) )  %>%
  summarize( percentage = 100 * n() / nrow(Xfraud) ) %>% round(., digits = 2)


## ----amount plot, echo = FALSE, warning = FALSE, fig.cap= 'Dispersion of transactions over amount'----
#plot a custom graph with data from X and Y
ggplot(data = X, mapping = aes(factor(Y), amount, color = factor(type))) +
  geom_jitter(alpha = 0.4, size = 0.85) +
  scale_x_discrete(labels = c('Legitimate', 'Fraud')) + # encode the x label
  #scale_y_log10() + # transform with log10 the y scale
  scale_color_discrete(breaks = c(0,1), labels = c('TRANSFER', 'CASH_OUT')) + # encode the legend
  labs(color = 'Transaction:', y = 'Amount') + xlab(NULL) + # display 'Type', hide name of x axis
  guides(color = guide_legend(override.aes = list(size = 3))) # Display larger colors in the legend
  + ggtitle(label = 'Amount plot') # Show title


## ----step plot, echo = FALSE, warning = FALSE, fig.cap='Dispersion of transactions over step (time)'----
#plot a custom graph with data from X and Y
ggplot(data = X, mapping = aes(factor(Y), step, color = factor(type))) +
  geom_jitter(alpha = 0.4, size = 0.085) +
  scale_x_discrete(labels = c('Legitimate', 'Fraud')) + # encode the x label
  scale_color_discrete(breaks = c(0,1), labels = c('TRANSFER', 'CASH_OUT')) + # encode the legend
  labs(color = 'Transaction:', y = 'Step (number of hours)') + xlab(NULL) + # display 'Type', hide name of x axis
  guides(color = guide_legend(override.aes = list(size = 3))) +  # Display larger colors in the legend
  ggtitle(label = 'Dispersion of balance_error_dest') # Show title


## ----balance_error_orig, fig.cap='Balance error origin account', echo = FALSE, warning = FALSE------
#plot a custom graph with data from X and Y
ggplot(data = X, mapping = aes(factor(Y), balance_error_orig, color = factor(type))) +
  geom_jitter(alpha = 0.4, size = 0.085) +
  scale_x_discrete(labels = c('Legitimate', 'Fraud')) + # encode the x label
  scale_color_discrete(breaks = c(0,1), labels = c('TRANSFER', 'CASH_OUT')) + # encode the legend
  labs(color = 'Transaction:', y = 'Balance error origin account') + xlab(NULL) + # display 'Type', hide name of x axis
  guides(color = guide_legend(override.aes = list(size = 3))) + # Display larger colors in the legend
  ggtitle(label = 'Dispersion of balance_error_dest') # Show title


## ----balance_error_dest, echo = FALSE, warning = FALSE, fig.cap='Balance error destination account'----
#plot a custom graph with data from X and Y
ggplot(data = X, mapping = aes(factor(Y), balance_error_dest, color = factor(type))) +
  geom_jitter(alpha = 0.4, size = 0.85) +
  scale_x_discrete(labels = c('Legitimate', 'Fraud')) + # encode the x label
  scale_color_discrete(breaks = c(0,1), labels = c('TRANSFER', 'CASH_OUT')) + # encode the legend
  labs(color = 'Transaction:', y = 'Balance error destination account') + xlab(NULL) + # display 'Type', hide name of x axis
  guides(color = guide_legend(override.aes = list(size = 3))) + # Display larger colors in the legend
  ggtitle(label = 'Dispersion of balance_error_dest') # Show title


## ----xfraud correlations----------------------------------------------------------------------------
# compute the correlations ignoring the NA's: 'complete.obs'
correlation_nonfraud = cor(Xnonfraud[,names(Xnonfraud) != 'step'], use = 'complete.obs')
correlation_fraud = cor(Xfraud[,names(Xfraud) != 'step'], use = 'complete.obs')


## ---- fig.cap= 'Heatmaps of correlation of transactions', echo = FALSE, message = FALSE, warning = FALSE----
# hide package loading messages 
suppressPackageStartupMessages(library(ComplexHeatmap))
# Create heatmap for fraud and non-fraud cases, using the ComplexHeatmap package
f <- Heatmap(correlation_fraud, column_title = 'Fraud', 
             row_order = names(correlation_nonfraud),     # columns and
             column_order = names(correlation_nonfraud),  # rows in the same order
             show_heatmap_legend = FALSE,    # Same legend for both, only display one
             rect_gp = gpar(col = 'white', lwd = 3))
nf <- Heatmap(correlation_nonfraud, column_title = 'Non-fraud',  
              row_order = names(correlation_nonfraud),     # columns and
              column_order = names(correlation_nonfraud),  # rows in the same order
              name = 'Coeficient',    # set the legend name
              rect_gp = gpar(col = 'white', lwd = 3))  # adjust display

# arrange the heatmaps side by side
draw(nf + f, heatmap_legend_side = 'left')

# Remove objects from memory
rm(Xfraud, Xnonfraud)


## ----prepare svm data, echo = FALSE, warning = FALSE, message = FALSE-------------------------------
# Join X and Y to make a balanced split of the data based on 
# classes for validation set, later for train and test sets. 
d <- cbind(X,isFraud = Y)
d <- d %>% mutate(step = as.numeric(step),
                 type = as.numeric(type),
                 isFraud = as.numeric(isFraud))

# # display the structure of 'd'
#str(d)

# Setting a fixed seed to enable reproducibility of the results.
set.seed(1, sample.kind = 'Rounding') # if using R 3.5 or earlier, use `set.seed(1)`

# encoding the target feature 'isFraud' as a factor
svm_d <- d %>% mutate(            
            isFraud = factor( isFraud, levels = c(0,1) ) )

# remove from memory
rm(d)

########################################################################
# # if your computer keeps crashing you can reduce the number of features here:
# # reduce the number of features to test the svm algoritm
# features_to_keep <- c('amount', 'type', 'newbalanceOrig', 'oldbalanceOrig', 'isFraud')
# 
# # drop the unwanted columns
# svm_d <- svm_d[, names(svm_d) %in% features_to_keep]
########################################################################


# set seed to 1 replicate the results
set.seed(1, sample.kind = 'Rounding')

# Reduce the number of observations to be proportional between the two classes
# to the isFraud == 1 minority class. (downSample function, Caret package)
svm_d  <- downSample(x = svm_d[ , -ncol(svm_d) ],
                    y = svm_d[ , ncol(svm_d) ],
                    yname = 'isFraud', list = FALSE)

# display the distribution of the classes in the new data
dplyr::count(svm_d, isFraud, sort = TRUE) %>% kable(caption='Output of the downSample function')

# show the features used to learn the model
str( svm_d[ , -ncold( svm_d ) ] )

# Create a index with 10 % of the raw PaySim data
svm_validation_index <- createDataPartition(y = svm_d$isFraud, times = 1, p = 0.1, list = FALSE)

# use the index to create the validation set and validation labels
svm_validation_set <- svm_d[ svm_validation_index, ]
# use the opposite of the index to create our dataset with 90% of the raw data
svm_paysim <- svm_d[ -svm_validation_index, ]

# In order to replicate the results here, you need to set the seed to 1
set.seed(1, sample.kind = 'Rounding')

#svm_paysim is the object we will use to learn our ML model, we start by creating the train and test sets
# Create a index for the test set
test_index <- createDataPartition(y = svm_paysim$isFraud, times = 1, p = 0.2, list = FALSE)

# Create the sets using the index, 20% of the PaySim data for testing, train data is 80%
train <- svm_paysim[ -test_index, ] # here we have the inverse of the index
test <- svm_paysim[ test_index, ] 


## ----SVM linear, echo = FALSE, warning = FALSE, message = FALSE-------------------------------------
# train model SVM - kernel: Linear'

# In order to replicate the results here, you need to set the seed to 1
set.seed(1, sample.kind = 'Rounding')

# train model SVM
model_svm <- svm( formula = isFraud ~ ., 
                 type = 'C-classification',
                 data= train,
                 kernel = 'linear' )

# make predictions on the test data
pred <- predict( model_svm, newdata = test[, -ncol(svm_d) ] ) 

# Confusion matrix
CM_SVM <- confusionMatrix( pred, test[,ncol(svm_d)], positive = '1' )

# Create table to store the results
results <- tibble(model = 'SVM', kernel = 'linear', data = 'test_set',
                  accuracy = CM_SVM$overall[1], sensitivity = CM_SVM$byClass[1],
                  specificity = CM_SVM$byClass[1])


## ---------------------------------------------------------------------------------------------------
# Confusion matrix for the SVM linear model on test set
CM_SVM


## ----SVM polynomial best tune, echo = FALSE, warning = FALSE, message = FALSE-----------------------
# Tunning of model SVM - kernel: polynomial')
# In order to replicate the results here, you need to set the seed to 1
set.seed(1, sample.kind = 'Rounding')

# # Long processing time here(findings bellow)
# # tune svm polynomial - function tune.svm()
# optimal_parameters <- tune.svm( x = train[,-ncol(svm_d)], y = train[,ncol(svm_d)],
#                           type = 'C-classification',
#                           kernel = 'polynomial', degree = 2, 
#                           cost = 10^(1:3), # range from 10 - 1000
#                           gamma = c(0.1, 1, 10), 
#                           coef0 = c(0.1, 1, 10) )
# optimal_parameters   

### best parameters:
###  degree gamma coef0 cost
###       2     1     1 1000

# In order to replicate the results here, you need to set the seed to 1
set.seed(1, sample.kind = 'Rounding')
# train model SVM
model_svm_poly <- svm( formula = isFraud ~ ., 
               data=train,
                kernel = 'polynomial',
                degree = 2, 
                gamma = 1,
                coef0 = 1,
                cost = 1000,
                scale = TRUE )

# Make predictions on the test data
pred_svm_poly <- predict(model_svm_poly, newdata = test[,-ncol(svm_d)])  # make predictions with test data
# Confusion matrix
CM_SVM_polynomial <- confusionMatrix(pred_svm_poly, test[,ncol(svm_d)], positive = '1') # show the confusion matrix
# Add results to table
results <- rbind(results, 
                 tibble(model = 'SVM', kernel = 'polynomial', data = 'test_set', 
                        accuracy = CM_SVM_polynomial$overall[1], sensitivity = CM_SVM_polynomial$byClass[1], 
                        specificity = CM_SVM_polynomial$byClass[1]))


## ----SVM validation performance, echo = FALSE, warning = FALSE, message = FALSE---------------------
# testing performance on VALIDATION data

# SVM linear - make predictions on validation set
pred_valid <- predict(model_svm, newdata = svm_validation_set[,-ncol(svm_d)])  # make predictions with validation data
# Confusion matrix
CM_SVM_val <- confusionMatrix(pred_valid, svm_validation_set[,ncol(svm_d)], positive = '1') # the confusion matrix
# Add results to table
results <- rbind(results, 
                 tibble(model = 'SVM', kernel = 'linear', data = 'validation_set', 
                        accuracy = CM_SVM_val$overall[1], sensitivity = CM_SVM_val$byClass[1], 
                        specificity = CM_SVM_val$byClass[1]))

# SVM polynomial - make predictions on validation set
pred_valid_poly <- predict(model_svm_poly, newdata = svm_validation_set[,-ncol(svm_d)])  

# Confusion matrix
CM_SVM_val_poly <- confusionMatrix(pred_valid_poly, svm_validation_set[,ncol(svm_d)], 
                              positive = '1') # the confusion matrix
# Add results to table
results <- rbind(results, 
                tibble(model = 'SVM', kernel = 'polynomial', data = 'validation_set', 
                accuracy = CM_SVM_val_poly$overall[1], sensitivity = CM_SVM_val_poly$byClass[1], 
                specificity = CM_SVM_val_poly$byClass[1]))

# display table with results
results %>% kable(caption = 'SVM Results')


## ----CM SVM polynomial------------------------------------------------------------------------------
# Confusion matrix for the SVM polynomial model on test set
CM_SVM_val_poly


## ----create validation set at 10% ratio, echo = FALSE, message = FALSE, warning=FALSE---------------
# A way to improve the ML training would be to replace these values with NA.
# replacing with NA's where there is a high chance of fraud
# filter the dataframe, then select the columns to update, lastly define the desired value (NA)
X[which( X$oldbalanceOrig == 0 & X$newbalanceOrig == 0 & X$amount != 0 ), 
  names(X) %in% c('oldbalanceOrig','newbalanceOrig') ] <- NA


# Setting a fixed see to enable reproducibility of the results.
# Join X and Y to make a balanced split of the data for validation set. 
d <- cbind( X, isFraud = Y )

# Force all features to numeric type.
d <- d %>% mutate(step = as.numeric(step),
                 type = as.numeric(type),
                 isFraud = as.numeric(isFraud))

# Setting a fixed seed to enable reproducibility of the results.
set.seed(1, sample.kind = 'Rounding') # if using R 3.5 or earlier, use `set.seed(1)`

# Create a index with 10 % of the raw PaySim data
validation_index <- createDataPartition(y = d$isFraud, times = 1, p = 0.1, list = FALSE)

# use the index to create the validation set and validation labels
validation_set <- d[ validation_index, ]
# use the opposite of the index to create our dataset with 90% of the raw data
paysim <- d[ -validation_index, ]

# Optimize resources by removing objects from memory
rm(d)


## ----Building the train and test sets, echo = FALSE, message = FALSE, warning=FALSE-----------------
# In order to replicate the results here, you need to set the seed to 1.
set.seed(1, sample.kind = 'Rounding')

# Create a index for the test set
test_index <- createDataPartition(y = paysim$isFraud, times = 1, p = 0.2, list = FALSE)
# Create the sets using the index, 20% of the PaySim data for testing, train data is 80%
train_set <- paysim[ -test_index, ] # here we have the inverse of the index 
test_set <- paysim[ test_index, ] 


## ---------------------------------------------------------------------------------------------------
# Get all the data types from the columns in the data using sapply() function
sapply(paysim, class)


## ----train xgboost----------------------------------------------------------------------------------
### XGB MODEL

# Converting train and test into xgb.DMatrix format
Dtrain <- xgb.DMatrix(
        data = as.matrix(train_set[, !names(train_set) %in% c('isFraud')]), 
        label = train_set$isFraud)
Dtest <- xgb.DMatrix(
         data = as.matrix(test_set[, !names(test_set) %in% c('isFraud')]),
        label = test_set$isFraud)

# Model Building: XGBoost
param_list = list(
  objective = "binary:logistic",
  eval_metric = 'error', 
  eta = 1,
  gamma = 1,
  max_depth = 3,
  subsample = 0.8,
  colsample_bytree = 0.5)
  
# # 5-fold cross-validation to 
# # find optimal value of nrounds
# set.seed(1)  # Setting seed

# # Cross-validation to determine the best
# # parameter for nrounds
# xgbcv = xgb.cv(params = param_list, 
#                data = Dtrain, 
#                nrounds = 300, 
#                nfold = 5, 
#                print_every_n = 10, 
#                early_stopping_rounds = 10, 
#                metrics = list('error'),
#                maximize = F)

# xgbcv

# In order to replicate the results here, you need to set the seed to 1.
set.seed(1, sample.kind = 'Rounding')

# Training XGBoost model at nrounds = 100
xgb_model = xgb.train(data = Dtrain, 
                      params = param_list, 
                      nrounds = 21)   # 21 best value from Cross-Valid


## ----xgboost model info, echo = TRUE----------------------------------------------------------------
# Display Xgb model info
xgb_model


## ----importance xgboost, echo = FALSE, warning = FALSE, message = FALSE, fig.cap='XGBoost feature importance plot'----
# Name of the features used by the model
names <- dimnames(Dtrain)[[2]]

# Importance of the features (variables) for the model
var_imp = xgb.importance( feature_names = names, 
             model = xgb_model)

# Plot the variable importance
xgb.plot.importance(var_imp)


## ----tree xgboost, include = FALSE, echo = FALSE, fig.cap='Structure of the first tree of the XGBoost model'----
# # Display the first tree
# xgb_tree <- xgb.plot.tree(model = xgb_model, trees = 0, 
#                           show_node_id = TRUE, render = FALSE)
# # # show the first tree of the model
# xgb_tree
# library(DiagrammeR)
# #export to pdf
# export_graph(xgb_tree, 'tree.pdf')
# knitr::include_graphics('tree.pdf')


## ----make predictions, warning = FALSE, message = FALSE, echo = FALSE-------------------------------

################## TEST SET ##################
# Make predictions on test data
predictions <- predict(xgb_model, Dtest)

# Compute classification error
test_error <- mean(as.numeric(predictions > 0.5) != test_set$isFraud)
print(paste('Test error: ', format(test_error*100, scientific = FALSE, digits = 5), '%'))

xgb_test <- mean(as.numeric(predictions > 0.5) == test_set$isFraud)

################################################################
# Calculate the Confusion Matrix (validation set)
pred_xgb <- ifelse(predictions > 0.5, 1, 0)
pred_xgb <- factor(pred_xgb, levels = c(0,1))
CM_xgb <- confusionMatrix(pred_xgb,   # predictions from model
                          factor(test_set$isFraud,  # labels to compare
                          levels = c(0,1)),  # levels of the label
                          positive = '1')   # positive variable

# Add results to table
results <- rbind(results, 
                tibble(model = 'XGBoost', kernel = 'binary:logistic', data = 'test_set', 
                accuracy = CM_xgb$overall[1], sensitivity = CM_xgb$byClass[1], 
                specificity = CM_xgb$byClass[1]))




################### VALIDATION SET ###############################
# validation set performance:
Val_data <- xgb.DMatrix(
         data = as.matrix(validation_set[, !names(validation_set) %in% c('isFraud')]),
            label = validation_set$isFraud)

# Make predictions on validation data
val_pred <- predict(xgb_model, Val_data)

# Compute classification error
val_error <- mean( as.numeric(val_pred > 0.5) != validation_set$isFraud )
print(paste('Validation error', format(val_error*100, scientific = FALSE, digits = 5), '%'))

xgb_accuracy <- mean(as.numeric(val_pred > 0.5) == validation_set$isFraud)

################################################################
# Calculate the Confusion Matrix (validation set)
pred_xgb_val <- ifelse(val_pred > 0.5, 1, 0)
pred_xgb_val <- factor(pred_xgb_val, levels = c(0,1))
CM_xgb_val <- confusionMatrix(pred_xgb_val,      # predictions from model validations set
                              factor(validation_set$isFraud, # actual values to compare
                                     levels = c(0,1)),   # levels of the legend
                              positive = '1')     # the positive class

# Add results to table
results <- rbind(results, 
                tibble(model = 'XGBoost', kernel = 'binary:logistic', data = 'validation_set', 
                accuracy = CM_xgb_val$overall[1], sensitivity = CM_xgb_val$byClass[1], 
                specificity = CM_xgb_val$byClass[1]))

# table with only xgb results:
xgb_results <- results[6:7,] %>% kable(caption = 'XGBoost Results')

# table with all results
all_results <- results %>% kable(caption = 'SVM and XGBoost Results')


## ----confusion matrix xgb validation set, include = FALSE-------------------------------------------
# Show confusion matrix XGB on validation set
CM_xgb_val


## ----Table XGBoost Results, echo = FALSE------------------------------------------------------------
print('Table XGBoost Results')
xgb_results


## ----Table with all results-------------------------------------------------------------------------
# display table with all results
all_results

