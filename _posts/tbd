---
title: 'Oversampling for Time Series Forecasting'
date: 2021-01-08 00:04:13
featured_image: '/images/0005720_coming-soon-page_550.jpeg'
excerpt: Coming soon...
---

### BACKGROUND
This project is intended to explore the concept of Oversampling, as it pertains to Time Series Forecasting. Time Series forecasting has been a success over the years because models are trained to recognize different sequences of data, to make predictions. At the surface level, we know that Deep Learning models rely on huge amounts of data to learn effectively and data and recognize the different patterns represented in the data. But what happens when we don't have enough data? Or what if we have a lot of data but it fails to represent every single, true observation well enough? Can we effectively oversample the Time Series data in hand, to result in a more representative dataset that enhances learning? For the purposes of this project, we can affirm that our Oversampling technique
is successful if the Mean Squared Error (MSE) calculated on our validation set is either lower or comparable to the MSE value before Oversampling was employed.

### OVERSAMPLING
Unlike Oversampling techniques for classification tasks, there is very little literature on Oversampling for Regression tasks. A simple reason for this is that by definition, if given 2 different categories, Oversampling is intended to increase the number of observations of whichever category has less observations, to reduce the effect of a class imbalance on our predictions. So, for example, if we were to classify images of animals as CAT or DOG and we have 5000 observations for cats but only 1300 observations for dog, we could oversample by increasing the number of observations of dogs to about the same number of observations as cats, so that the model has at least an almost equal number of observations to learn from. In the case of sequential data like we have for Time Series, however, the case for wanting to Oversample relies on 2 things:
• the basic understanding that Deep Learning models generally thrive off copious amounts of data.
• the need to ensure that every single observation in the dataset is well represented.

To emulate this definition of Oversampling directly, we could use Exploratory Data Analysis to create separate bins for different intervals of forecasts, to check and consequently increase the number of observations from a bin that is not represented enough times in the dataset. So for example, how many of the sales fall into the 1-10 range, 11-20 range, 30-40 range, and so on. If we observe that one of these bins does not have enough observations compared to the other bins, we could explore oversampling to increase the observations to a number close to the observations in the other bins. The possible problem with both of these ideas is the fact that in the process of Oversampling, the "minority" bin will have no oversight over the nature of the observations in the other bins, so we could end up with a result completely opposite to what is desired, i.e., an undesired result of having sequential data in the various bins being so similar that it becomes much more difficult for the model to learn. The most inspiring paper on this topic is that of Nuno Moniz, Paula Branco & Luís Torgo: RESAMPLING STRATEGIES FOR IMBALANCED TIME SERIES FORECASTING. It is currently the only paper on the topic.
(Link to paper: [https://link.springer.com/article/10.1007/s41060-017-0044-3#Sec3]

• The aforementioned paper explored different resampling strategies. The  authors defined resampling strategies as pre-processing approaches that change the original data distribution in order to meet some user-given criteria.
• According to the paper, "among the advantages of pre-processing strategies is the ability of using any standard learning tool" i.e., after resampling, I should be able to select and use any Machine Learning model I prefer. However, to match a change in the data distribution with the user preferences is
not a trivial task.
• Additionally, the paper focused on resampling strategies that aim at preprocessing the data for obtaining an increased predictive performance in cases that are scarce and simultaneously important to the user. In our case, what is of most importance is the sale value for any store of interest. Also, for the
purposes of this project and because of the nature of our dataset, we can define "scarce" as an observation where the sale value is an outlier, i.e., if we were to observe a sale value in the 100s or high 10s, what would its sequential data look like?

We first have to check to see that this observation is credible, and not merely an error in entry. If this observation is
credible, how can we synthetically create more observations that look like it to increase its representation?
• The authors went on to describe importance: a relevance function 𝜙(𝑌). "Being domain-dependent information, it is the user responsibility to specify the relevance function. Nonetheless, when lacking expert knowledge, it is possible to automatically generate the relevance function. Being a continuous function on the scale [0, 1], we require the user to specify a relevance threshold, 𝑡𝑅, that establishes the minimum relevance score for a certain value of the target variable to be considered relevant. This threshold is only required because the proposed resampling strategies need to be able to decide which values are the most relevant when the distribution changes."

### MODEL SELECTION
Univariate LSTM models would be explored because of the nature of our dataset. In our dataset, we have a single series of observations and a model is required to learn from the series of past observations to predict the sale value of a future observation, when given a sequence. The initial plan was to explore several LSTM models, but further research showed that some of these are more specialized for certain tasks. For example, the CNN LSTM is generally used for problems that have a temporal structure in their input such as the order of images in a video or words in text or require the generation of output with temporal
structure such as words in a textual description. This makes it more suitable for image classification or Natural Language Processing tasks. Furthermore, the Oversampling process proved to be more difficult than anticipated as the authors of the paper did not make their code available to the public. For this reason, instead of trying Oversampled data on several LSTM models, my focus shifted to actually being able to implement an Oversampling method that makes sense for our dataset, as well as the problem we’re trying to solve.

### FINAL APPROACH
The Oversampling approach that best meet the needs of this project while simultaneously drawing inspiration from the aforementioned paper is Synthetic Minority Over-Sampling Technique for Regression with Gaussian Noise. This approach involves conducting the Synthetic Minority Over-Sampling Technique for Regression
(SMOTER) with traditional interpolation, as well as with the introduction of Gaussian Noise (SMOTERGN). It selects between the two over-sampling techniques by the KNN distances underlying a given observation:
• If the distance is close enough, SMOTER is applied.
• If too far away, SMOTER-GN is applied.

Further research proved that this method is useful for prediction problems where regression is applicable, but the values in the interest of predicting are rare or uncommon, as seen in the paper. It can also serve as a useful alternative to log transforming a skewed response variable, especially if generating synthetic data is also of interest. [Gaussian noise]. We can define it as the effect of adding lots of random things together. This ultimately results in a Probability Density Function that gives an idea of which values are more likely under the PDF curve, and ultimately, worth considering as appropriate for a synthetic observation. In other words, what we are doing here through this Oversampling Technique is: Selecting different but similar values from the observations at hand. Using the gaussian noise to select a different set of values Using KNN to determine which of these new, synthetic samples resemble the original, rare, observation more closely. If the distance is close enough, SMOTER is applied. If too far away, SMOTER-GN is applied Eventually, we have a dataset that tries to ensure that each sequence and its output, i.e., the sale value, are well represented in the entire dataset. 2 modeling approaches were employed:
• The Univariate LSTM model that used in Assignment 4 will be duplicated.
• Facebook's Time Series model, Prophet, which gave an impressive MSE of 4.71 will also be used/compared.

### RESULTS
So how do we determine if Oversampling works? I determined this comparing the RMSE values of the 2 non-tuned models to 2, tuned different models:
• The tuned LSTM model I built in Assignment 4
• A tuned version of Facebook's PROPHET model.

Reason: I believe that if oversampling truly works, we should be able to see a model that performs well in its raw form, and this performance should be comparable to an LSTM model that has been tuned on the normal data and is in its best version. The RMSE plots above are comparable to the plots of RMSE from tuned models that were created in the original assignment, as well as the tuned version of Facebook's Prophet model. Because of the random nature of the Oversampling process, the RMSE value changes whenever this notebook is rerun entirely. For example, we can expect to see RMSE values as low as 7.55 for the default, univariate model, and as low as 4.97 on the Prophet model (as presented to the class on Tuesday, 12/07). However, the key finding here is that these MSE values from the Oversampled data  are encouragingly low enough, and we can confidently expect further Tuning to help see much lower RMSE values.

#### RESULTS FROM LSTM MODEL
Before Oversampling, after Tuning
RMSE: 7.35
After Oversampling, before Tuning
RMSE: 7.59

#### RESULTS FROM PHOPHET MODEL
Before Oversampling, after Tuning
RMSE: 4.72
After Oversampling, before Tuning
RMSE: 4.97

#### TABLE OF RESULTS
Univariate
LSTM Model
Facebook’s
PROPHET
MSE (without
Oversampling
after Tuning)
7.35 4.71
MSE (after
Oversampling,
before Tuning)
7.56 4.97

### LIMITATIONS/NEXT STEPS
The lack of literature on the subject is prove that this topic has not been researched enough. This consequently leads to a lack of benchmarks to compare my findings to. However, I can continue to make breakthroughs by ensuring that subsequent Oversampling techniques I create are compared on the same scale to the Techniques created in this notebook. By so doing, I can continue to make contributions to the field, as well as continue to ensure that people who explore my findings are confident in them.

###CONCLUSION
Hopefully, this project proves that Oversampling can be applied to sequential data, especially data that pertains to Time Series Forecasting. I have thoroughly enjoyed exploring this idea from its inception to the final implementation. I hope it serves you well. Thank you.

CREDITS
• Nick Kunz
• Nuno Moniz, Paula Branco & Luís Torgo
• Yuankai Huo, Yi Zuo & Ruining Deng
BIBLIOGRAPHY
• Nickkunz. “Nickkunz/Smogn.” GitHub,
github.com/nickkunz/smogn.
• O. Akbilgic, H. Bozdogan, et al.
“Resampling Strategies for Imbalanced
Time Series Forecasting.” International
Journal of Data Science and Analytics,
Springer International Publishing, 1 Jan.
1970,
link.springer.com/article/10.1007/s41060-
017-0044-3.
• Paobranco. “Paobranco/SMOGNLIDTA17.” GitHub,
github.com/paobranco/SMOGN-LIDTA17.
• “Prophet Is a Forecasting Procedure
Implemented in R and Python. It Is Fast
and Provides Completely Automated
Forecasts That Can Be Tuned by Hand by
Data Scientists and Analysts.” Prophet,
facebook.github.io/prophet/.
