# Rubix ML - Text Sentiment Analyzer
This is a multilayer feed forward neural network for text sentiment classification trained on 25,000 movie reviews from the [IMDB](https://www.imdb.com/) movie reviews website. The dataset also provides another 25,000 samples which we use after training to test the model. This example project demonstrates text feature representation and deep learning in Rubix ML using a neural network classifier called a Multilayer Perceptron.

- **Difficulty:** Hard
- **Training time:** Hours
- **Memory required:** 12G

## Installation
Clone the project locally with [Git](https://git-scm.com/):
```sh
$ git clone https://github.com/RubixML/Sentiment
```

> **Note:** Cloning may take longer than usual because of the large dataset.

Install project dependencies with [Composer](http://getcomposer.org/):
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.2 or above

## Tutorial

### Introduction
Our objective is to predict the sentiment (either *positive* or *negative*) of a blob of English text using machine learning. We sometimes refer to this type of ML as [Natural Language Processing](https://en.wikipedia.org/wiki/Natural_language_processing) (NLP) because it involves machines making sense of language. The dataset provided to us contains 25,000 training and 25,000 testing samples each consisting of a blob of English text reviewing a movie on the IMDB website. The samples have been labeled positive or negative based on the score (1 - 10) the reviewer gave to the movie. From there, we'll use the IMDB dataset to train a multilayer neural network to predict the sentiment of any English text we show it.

**Example**

> "Story of a man who has unnatural feelings for a pig. Starts out with a opening scene that is a terrific example of absurd comedy. A formal orchestra audience is turned into an insane, violent mob by the crazy chantings of it's singers. Unfortunately it stays absurd the WHOLE time with no general narrative eventually making it just too off putting. ..."

### Extracting the Data
The samples are given to us in individual `.txt` files and organized by label into `pos` and `neg` folders. We'll use PHP's built in `glob()` function to loop through all the text files in each folder and add their contents to a samples array. We'll also add the corresponding *positive* and *negative* labels in their own array.

> **Note**: The source code for this example can be found in the [train.php](https://github.com/RubixML/Sentiment/blob/master/train.php) file in the project root.

```php
$samples = $labels = [];

foreach (glob('train/pos/*.txt') as $file) {
    $samples[] = [file_get_contents($file)];
    $labels[] = 'positive';
}

foreach (glob('train/neg/*.txt') as $file) {
    $samples[] = [file_get_contents($file)];
    $labels[] = 'negative';
}
```

Now, we can instantiate a new [Labeled](https://docs.rubixml.com/en/latest/datasets/labeled.html) dataset object with the imported samples and labels.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = new Labeled($samples, $labels);
```

### Dataset Preparation
Neural networks compute a non-linear continuous function and therefore require continuous features as inputs. However, the samples given to us in the IMDB dataset are in raw text format. Therefore, we'll need to convert those text blobs to continuous features before training. We'll do so using the [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model) technique which produces long sparse vectors of word counts using a fixed vocabulary. The entire series of transformations necessary to prepare the incoming dataset for the network can be implemented in a transformer [Pipeline](https://docs.rubixml.com/en/latest/pipeline.html).

First, we'll apply an [HTML Stripper](https://docs.rubixml.com/en/latest/transformers/html-stripper.html) to sanitize the text from any unimportant structure or formatting markup, just in case. Then [Text Normalizer](https://docs.rubixml.com/en/latest/transformers/text-normalizer.html) will convert all characters to lowercase and remove any extra whitespace. The [Word Count Vectorizer](https://docs.rubixml.com/en/latest/transformers/word-count-vectorizer.html) is responsible for creating a continuous feature vector of word counts from the raw text and [TF-IDF Transformer](https://docs.rubixml.com/en/latest/transformers/tf-idf-transformer.html) applies a weighting scheme to those counts. Finally, [Z Scale Standardizer](https://docs.rubixml.com/en/latest/transformers/z-scale-standardizer.html) takes the TF-IDF weighted counts and centers and scales the sample matrix to have 0 mean and unit variance. This last step will help the neural network converge quicker.

The Word Count Vectorizer is a bag-of-words feature extractor that uses a fixed vocabulary and term counts to quantify the words that appear in a particular document. We elect to limit the size of the vocabulary to 10,000 of the most frequent words that satisfy the criteria of appearing in at least 3 different documents. In this way, we limit the amount of *noise* words that enter the training set.

Another common feature representation for words are their [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) values which take the term frequencies (TF) from Word Count Vectorizer and weight them by their inverse document frequencies (IDF). IDFs can be interpreted as the word's *importance* within the text corpus. Specifically, higher weight is given to words that are more rare within the corpus.

### Instantiating the Learner
The next thing we'll do is define the architecture of the neural network and instantiate the [Multilayer Perceptron](https://docs.rubixml.com/en/latest/classifiers/multilayer-perceptron.html) classifier. The network uses 5 hidden layers consisting of a [Dense](https://docs.rubixml.com/en/latest/neural-network/hidden-layers/dense.html) layer of neurons followed by a non-linear [Activation](https://docs.rubixml.com/en/latest/neural-network/hidden-layers/activation.html) layer and an optional [Batch Norm](https://docs.rubixml.com/en/latest/neural-network/hidden-layers/batch-norm.html) layer for normalizing the activations. The first 3 hidden layers use a [Leaky ReLU](https://docs.rubixml.com/en/latest/neural-network/activation-functions/leaky-relu.html) activation function while the last 2 utilize a trainable form of the Leaky ReLU called [PReLU](https://docs.rubixml.com/en/latest/neural-network/hidden-layers/prelu.html) for *Parametric* Rectified Linear Unit. The benefit that *leakage* provides over standard rectification is that it allows neurons to learn even if they did not activate by allowing a small gradient to pass through during backpropagation. We've found that this architecture works fairly well for this problem but feel free to experiment on your own.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\HTMLStripper;
use Rubix\ML\Transformers\TextNormalizer;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Other\Tokenizers\NGram;
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel(
    new Pipeline([
        new HTMLStripper(),
        new TextNormalizer(),
        new WordCountVectorizer(10000, 3, new NGram(1, 2)),
        new TfIdfTransformer(),
        new ZScaleStandardizer(),
    ], new MultiLayerPerceptron([
        new Dense(100),
        new Activation(new LeakyReLU()),
        new Dense(100),
        new Activation(new LeakyReLU()),
        new Dense(100),
        new BatchNorm(),
        new Activation(new LeakyReLU()),
        new Dense(50),
        new PReLU(),
        new Dense(50),
        new PReLU(),
    ], 200, new AdaMax(0.0001))),
    new Filesystem('sentiment.model', true)
);
```

We'll choose a batch size of 200 samples and perform network parameter updates using the [AdaMax](https://docs.rubixml.com/en/latest/neural-network/optimizers/adamax.html) optimizer. AdaMax is based on the [Adam](https://docs.rubixml.com/en/latest/neural-network/optimizers/adam.html) algorithm but tends to handle sparse updates better. When setting the learning rate of an optimizer, the important thing to note is that a learning rate that is too low will cause the network to learn slowly while a rate that is too high will prevent the network from learning at all. A global learning rate of 0.0001 seems to work pretty well for this problem.

Lastly, we'll wrap the entire estimator in a [Persistent Model](https://docs.rubixml.com/en/latest/persistent-model.html) wrapper so we can save and load it later in our validation script. The [Filesystem](https://docs.rubixml.com/en/latest/persisters/filesystem.html) persister object tells the wrapper to save and load the serialized model data from a path on disk. Setting the history parameter to true tells the persister to keep a history of past saves.

### Training
Now, you can call the `train()` method with the training dataset we instantiated earlier to start the training process.

```php
$estimator->train($dataset);
```

### Validation Score and Loss
During training, the learner will record the validation score and the training loss at each epoch. The validation score is calculated using the default [F Beta](https://docs.rubixml.com/en/latest/cross-validation/metrics/f-beta.html) metric on a hold out portion of the training set. Contrariwise, the training loss is the value of the cost function (in this case the [Cross Entropy](https://docs.rubixml.com/en/latest/neural-network/cost-functions/cross-entropy.html) loss) computed over the training data. We can visualize the training progress by plotting these metrics. To export the scores and losses you can call the additional `scores()` and `steps()` methods respectively.

```php
$scores = $estimator->scores();

$losses = $estimator->steps();
```

Here is an example of what the validation score and training loss looks like when they are plotted. The validation score should be getting better with each epoch as the loss decreases. You can generate your own plots by importing the `progress.csv` file into your favorite plotting software.

![F1 Score](https://raw.githubusercontent.com/RubixML/Sentiment/master/docs/images/validation-score.svg?sanitize=true)

![Cross Entropy Loss](https://raw.githubusercontent.com/RubixML/Sentiment/master/docs/images/training-loss.svg?sanitize=true)

### Saving
Finally, we save the model so we can load it later in our other scripts.

```php
$estimator->save();
```

### Cross Validation
To test the generalization performance of the trained network we'll use the testing samples provided to us to generate predictions and then analyze them compared to their ground-truth labels using a cross validation report. Note that we do not use any training data for cross validation because we want to test the model on samples it has never seen before.

> **Note**: The source code for this example can be found in the [validate.php](https://github.com/RubixML/Sentiment/blob/master/validate.php) file in the project root.

We'll start by importing the testing samples from the `test` folder like we did with the training samples.

```php
$samples = $labels = [];

foreach (glob('test/pos/*.txt') as $file) {
    $samples[] = [file_get_contents($file)];
    $labels[] = 'positive';
}

foreach (glob('test/neg/*.txt') as $file) {
    $samples[] = [file_get_contents($file)];
    $labels[] = 'negative';
}
```

Then, load the samples and labels into a [Labeled](https://docs.rubixml.com/en/latest/datasets/labeled.html) dataset object using the `build()` method, randomize the order, and take the first 10,000 rows and put them in a new dataset object.

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::build($samples, $labels)->randomize()->take(10000);
```

Next, we'll use the Persistent Model wrapper to load the network we trained earlier.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('sentiment.model'));
```

Now we can use the estimator to make predictions on the testing set. The `predict()` method takes a dataset as input and returns an array of predictions from the model.

```php
$predictions = $estimator->predict($dataset);
```

The cross validation report we'll generate is actually a combination of two reports - [Multiclass Breakdown](https://docs.rubixml.com/en/latest/cross-validation/reports/multiclass-breakdown.html) and [Confusion Matrix](https://docs.rubixml.com/en/latest/cross-validation/reports/confusion-matrix.html). We wrap each report in an [Aggregate Report](https://docs.rubixml.com/en/latest/cross-validation/reports/aggregate-report.html) to generate both reports at once. The Multiclass Breakdown will give us detailed information about the performance of the estimator at the class level. The Confusion Matrix will give us an idea as to what labels the estimator is *confusing* one another for.

```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);
```

To generate the report, pass in the predictions along with the labels from the testing set to the `generate()` method on the report.

```php
$results = $report->generate($predictions, $dataset->labels());
```

Take a look at the report and see how well the model performs. According to the example report below, our model is 87% accurate.

```json
[
    {
        "overall": {
            "accuracy": 0.8732,
            "precision": 0.8731870662254005,
            "recall": 0.8732522811796086,
            "specificity": 0.8732522811796086,
            "negative_predictive_value": 0.8731870662254005,
            "false_discovery_rate": 0.12681293377459946,
            "miss_rate": 0.12674771882039138,
            "fall_out": 0.12674771882039138,
            "false_omission_rate": 0.12681293377459946,
            "f1_score": 0.8731922850186206,
            "mcc": 0.7464393445561573,
            "informedness": 0.7465045623592172,
            "markedness": 0.7463741324508011,
            "true_positives": 8732,
            "true_negatives": 8732,
            "false_positives": 1268,
            "false_negatives": 1268,
            "cardinality": 10000
        },
        "label": {
            "positive": {
                "accuracy": 0.8732,
                "precision": 0.8673080777710964,
                "recall": 0.8771538617474154,
                "specificity": 0.8693507006118019,
                "negative_predictive_value": 0.8790660546797047,
                "false_discovery_rate": 0.13269192222890358,
                "miss_rate": 0.12284613825258461,
                "fall_out": 0.13064929938819814,
                "false_omission_rate": 0.12093394532029533,
                "f1_score": 0.8722031848417658,
                "informedness": 0.7465045623592172,
                "markedness": 0.7463741324508011,
                "mcc": 0.7464393445561573,
                "true_positives": 4327,
                "true_negatives": 4405,
                "false_positives": 662,
                "false_negatives": 606,
                "cardinality": 4933,
                "density": 0.4933
            },
            "negative": {
                "accuracy": 0.8732,
                "precision": 0.8790660546797047,
                "recall": 0.8693507006118019,
                "specificity": 0.8771538617474154,
                "negative_predictive_value": 0.8673080777710964,
                "false_discovery_rate": 0.12093394532029533,
                "miss_rate": 0.13064929938819814,
                "fall_out": 0.12284613825258461,
                "false_omission_rate": 0.13269192222890358,
                "f1_score": 0.8741813851954753,
                "informedness": 0.7465045623592172,
                "markedness": 0.7463741324508011,
                "mcc": 0.7464393445561573,
                "true_positives": 4405,
                "true_negatives": 4327,
                "false_positives": 606,
                "false_negatives": 662,
                "cardinality": 5067,
                "density": 0.5067
            }
        }
    },
    {
        "positive": {
            "positive": 4327,
            "negative": 662
        },
        "negative": {
            "positive": 606,
            "negative": 4405
        }
    }
]
```

Nice job! Now you're ready to make predictions on some new data.

### Predicting Single Samples
Now that we're confident with our model, let's build a simple script that takes some text input from the terminal and outputs a sentiment prediction using the estimator we've just trained.

> **Note**: The source code for this example can be found in the [predict.php](https://github.com/RubixML/Sentiment/blob/master/predict.php) file in the project root.

First, load the model from storage using the static `load()` method on the Persistent Model meta-estimator and the Filesystem persister pointed to the file containing the serialized model data.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('sentiment.model'));
```

Next, we'll use the built-in PHP function `readline()` to prompt the user to enter some text that we'll store in a variable.

```php
while (empty($text)) $text = readline("Enter some text to analyze:\n");
```

To make a prediction on the text that was just entered, call the `predictSample()` method on the learner with the single sample. The `predictSample()` method returns a single prediction that we'll print out to the terminal.

```php
$prediction = $estimator->predictSample([$text]);

echo "The sentiment is: $prediction" . PHP_EOL;
```

**Output**

```sh
Enter some text to analyze: Rubix ML is the best
The sentiment is: positive
```

### Next Steps
Congratulations on completing the tutorial on text sentiment classification in Rubix ML using a Multilayer Perceptron. We recommend playing around with the network architecture and hyper-parameters on your own to get a feel for how they effect the final model. Generally, adding more neurons and layers will improve performance but training may take longer. In addition, a larger vocabulary size may also improve the model at the cost of additional computation during training and inference.

## Original Dataset
See DATASET_README. For comments or questions regarding the dataset please contact [Andrew Maas](http://www.andrew-maas.net).

### References
>- Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
