# Text Sentiment Analyzer

This is a multi layer feed forward neural network for text sentiment classification (*positive* or *negative*) trained on 25,000 movie reviews from the [IMDB](https://www.imdb.com/) movie reviews website. The dataset also provides another 25,000 samples which we use to validate the model. This example project demonstrates text feature representation and deep learning using a type of neural network classifier called a [Multi Layer Perceptron](https://github.com/RubixML/RubixML#multi-layer-perceptron).

- **Difficulty**: Hard
- **Training time**: Long
- **Memory needed**: > 8G

## Installation

Clone the repository locally:
```sh
$ git clone https://github.com/RubixML/Sentiment
```

> **Note**: Cloning may take longer than usual because of the large dataset.

Install dependencies:
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

## Tutorial
Our objective is to predict the sentiment (either *positive* or *negative*) of a blob of English text. We sometimes refer to this type of machine learning as Natural Language Processing (or *NLP* for short) because it involves making sense of language. The dataset provided to us contains 25,000 training and 25,000 testing samples each consisting of a blob of English text describing a movie review. The samples have been pre labeled as either positive or negative.

### Training
Before we dive into the architecture of the network, let's first take care of loading the provided dataset into a [Labeled](https://github.com/RubixML/RubixML#labeled) dataset object. The samples are given to us in single `.txt` files and organized into `pos` and `neg` folders. We'll use PHP's built in `glob()` function to loop through all the text files in each folder and add their contents to a samples array. We'll also add *positive* and *negative* labels as the training signal.

> **Note**: The full code can be found in the `train.php` file.

```php
use Rubix\ML\Datasets\Labeled;

$samples = $labels = [];

foreach (glob(__DIR__ . '/train/pos/*.txt') as $file) {
        $samples[] = [file_get_contents($file)];
        $labels[] = 'positive';
}

foreach (glob(__DIR__ . '/train/neg/*.txt') as $file) {
        $samples[] = [file_get_contents($file)];
        $labels[] = 'negative';
}

$training = Labeled::build($samples, $labels);
```

Since the features given to us are in raw text format, we'll need to convert them to continuous values in order for the neural net to understand them and train effectively. We do so with a transformer pipeline consisting of [HTML Stripper](https://github.com/RubixML/RubixML#html-stripper), [Text Normalizer](https://github.com/RubixML/RubixML#text-normalizer), [Word Count Vectorizer](https://github.com/RubixML/RubixML#word-count-vectorizer), and [TF-IDF Transformer](https://github.com/RubixML/RubixML#tf-idf-transformer). If you are unfamiliar with transformer pipelines see the [Credit Card Default](https://github.com/RubixML/Credit) tutorial using Logistic Regression and the Pipeline wrapper.

The Word Count Vectorizer is a common *bag of words* feature extractor that uses a fixed vocabulary and term counts to denote words that appear in a particular document. We elect to limit the vocabulary to 10,000 of the most frequent words that satisfy the criteria of appearing in at least 3 different documents. In this way, we limit the amount of *noise* words that enter the training set. Another common text feature representation are TF-IDF values which take the term counts from Word Count Vectorizer and weight them by their inverse document frequencies (IDFs) which can be interpreted as their *importance* within the corpus. Specifically, higher weight is given to words that are more rare within the corpus.

> **Note**: The word counts and TF-IDF feature representations for this example throw away all sentence structure. Despite that, the model works pretty well in practice. In the future you can consider using [N-Grams](https://github.com/RubixML/RubixML#n-gram) instead of single words to recover some structual information.

The next thing we need to do is define the architecture of the network's hidden layers as the first hyper-parameter of the Multi Layer Perceptron base estimator. Each of the 5 hidden layers consist of a [Dense](https://github.com/RubixML/RubixML#dense) layer of neurons and a non-linear [Activation](https://github.com/RubixML/RubixML#activation) layer with optional [Dropout](https://github.com/RubixML/RubixML#dropout) regularization. The first 3 layers use a [Leaky ReLU](https://github.com/RubixML/RubixML#leaky-relu) activation function while the last 2 use a parametric form of the Leaky ReLU called [PReLU](https://github.com/RubixML/RubixML#prelu) (for *Parametric* Rectified Linear Unit). We've found that this architecture works pretty well for this problem but feel free to experiment and come up with your own.

> **Note**: For this tutorial, the "depth" of the hidden layers is distinguished as the number of *weight* layers which include the five Dense layers.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Transformers\HTMLStripper;
use Rubix\ML\Transformers\TextNormalizer;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel(new Pipeline([
    new HTMLStripper(),
    new TextNormalizer(),
    new WordCountVectorizer(10000, 3),
    new TfIdfTransformer(),
], new MultiLayerPerceptron([
    new Dense(100),
    new Activation(new LeakyReLU(0.1)),
    new Dropout(0.2),
    new Dense(70),
    new Activation(new LeakyReLU(0.1)),
    new Dropout(0.2),
    new Dense(50),
    new Activation(new LeakyReLU(0.1)),
    new Dropout(0.2),
    new Dense(30),
    new PReLU(0.25),
    new Dense(10),
    new PReLU(0.25),
], 300, new Adam(0.00001), 1e-4),
    new Filesystem('sentiment.model')
);
```

Observe the general pattern to the hidden layers of the network. [Dense](https://github.com/RubixML/RubixML#dense) layers linearly transform the input, then an [Activation](https://github.com/RubixML/RubixML#activation) layer applies a non-linear transformation, and the process repeats. Optionally we add a mild [Dropout](https://github.com/RubixML/RubixML#dropout) to the first three layers as a method to prevent overfitting. For the activations we are using two different types of Activation layers with the [Leaky ReLU](https://github.com/RubixML/RubixML#leaky-relu) activation function. The last two hidden layers use a parametric form of the Leaky ReLU (called [PReLU](https://github.com/RubixML/RubixML#prelu)) that learns the optimal amount of *leakage* to apply during training. Refer to the API Reference on [hidden layers](https://github.com/RubixML/RubixML#hidden-layers) for further reading.

The remaining hyper-parameters *batch size*, *optimizer* with *learning rate*, and *l2 regularization* amount will now be set. Batch size determines the number of training samples to run through the network at one time. The Gradient Descent optimizer determines the step size for each parameter in the network and most optimizers allow you to set a *learning rate* which controls the master step size. When setting the learning rate of an Optimizer, the important thing to note is that a learning rate that is too low will train slowly while a rate that is too high will prevent the network from learning at all. Lastly, we apply a small l2 penalty to the weights in the network such to help prevent them from overfitting the training data. For the full list of hyper-parameters, check out the [Multi Layer Perceptron](https://github.com/RubixML/RubixML#multi-layer-perceptron) API reference.

Lastly, we wrap the entire Pipeline in a [Persistent Model](https://github.com/RubixML/RubixML#persistent-model) wrapper so we can save and load it later in a different process.

Now we're ready to train, but first let's set a logger instance so we can receive training updates through the console. Then call `train()` with the training set we instantiated earlier to train the neural network classifier.

```php
use Rubix\ML\Other\Loggers\Screen;
use League\Csv\Writer;

$estimator->setLogger(new Screen('sentiment'));

$estimator->train($training);

$writer = Writer::createFromPath(PROGRESS_FILE, 'w+');
$writer->insertOne(['loss', 'score']);
$writer->insertAll(array_map(null, $estimator->steps(), $estimator->scores()));
```

We also use the PHP League's [CSV Writer](https://csv.thephpleague.com/) to dump the loss and validation scores at each epoch during training to a CSV file for easy importing into a plotting application such as [Tableu](https://public.tableau.com/en-us/s/) or [Google Sheets](https://www.google.com/sheets/about/). Notice that the loss should go down and the validation score should go up as training progresses. If this is not the case, you may need to adjust the learning rate or other hyper-parameters.

![Cross Entropy Loss](https://github.com/RubixML/Sentiment/blob/master/docs/images/training-loss-score.png)

> **Note**: Due to the high dimensionality of the TF-IDF input vectors and the width of the hidden layers this network has over 1 million individual weight parameters. Training may take a long time, however you will be able to monitor its progress using the Screen logger or any other PSR-3 compatible logger in real time.

Finally, we prompt to save the model so we can use it later in our other scripts.

```php
$estimator->prompt();
```

To run the training script from the project root:
```sh
$ php train.php
```

or

```sh
$ composer train
```

### Prediction
Now we'll build a simple script that takes some text input from the terminal and outputs a sentiment prediction using the estimator we've just trained.

> **Note**: The full code can be found in the `predict.php` file.

To load the trained MLP classifier, we need to tell Persistent Model where the model is located in storage with a [Persister](https://github.com/RubixML/RubixML#persisters) object. Persisters can be thought of as the storage *driver* used to persist the model.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('sentiment.model'));
```

Next, we'll use the build in PHP function `readline()` to prompt the user to enter some text and put the single sample in an [Unlabeled](https://github.com/RubixML/RubixML#unlabeled) dataset object.

```php
use Rubix\ML\Datasets\Unlabeled;

$text = readline('Enter text to analyze: ');

$dataset = Unlabeled::build([$text]);
```

Finally, we pass the dataset to the `proba()` method on the estimator to return an array of class probability estimates per sample and dump the first one.

```php
$probabilities = $estimator->proba($dataset);

var_dump($probabilities[0]);
```

To run the prediction script from the project root:
```sh
$ php predict.php
```

or

```sh
$ composer predict
```

### Cross Validation
To test the generalization performance of the trained network we'll use the testing samples provided to us to generate predictions and then analyze them compared to their ground-truth labels with a cross validation (*CV*) report. We do not use any training data for cross validation because we want to test the model on data it has never seen before.

> **Note**: The full code can be found in the `validate.php` file.

We'll start by importing the testing samples like we did with the training samples. This time, however, we're only going to use a subset of the testing data to generate the report. After we build the dataset we call `randomize()` and `take()` to create a testing set containing 5000 random samples.

```php
use Rubix\ML\Datasets\Labeled;

$samples = $labels = [];

foreach (glob(__DIR__ . '/test/pos/*.txt') as $file) {
        $samples[] = [file_get_contents($file)];
        $labels[] = 'positive';
}

foreach (glob(__DIR__ . '/test/neg/*.txt') as $file) {
        $samples[] = [file_get_contents($file)];
        $labels[] = 'negative';
}

$testing = Labeled::build($samples, $labels)->randomize()->take(5000);
```

Again, we use the Persistent Model wrapper to load the network we trainied earlier and then use it to make predictions on the testing set. The `predict()` method takes the testing set as input and returns an array of class predictions (*positive* or *negative*).

> **Note**: Unlike the `proba()` method, which outputs the probability scores for each label, the `predict()` method only outputs the predicted class label.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('sentiment.model'));

$predictions = $estimator->predict($testing);
```

The last step is to generate the report and write it to a JSON file. The report we'll generate is actually a combination of two reports ([Multiclass Breakdown](https://github.com/RubixML/RubixML#multiclass-breakdown) and [Confusion Matrix](https://github.com/RubixML/RubixML#confusion-matrix)). We wrap each report in an [Aggregate Report](https://github.com/RubixML/RubixML#aggregate-report) such to generate all reports at once. The Multiclass Breakdown will give us detailed information about the performance of the estimator broken down by class. The Confusion Matrix will give us an idea as to what labels the estimator is "confusing" for another. See the [API Reference](https://github.com/RubixML/RubixML#reports) for more information.

```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $testing->labels());

file_put_contents('report.json', json_encode($results, JSON_PRETTY_PRINT));
```

Now take a look at the report file in your favorite editor and see how well it performed. Our tests using the network architecture in this tutorial scores about 85% accurate. See if you can score higher by tuning the hyper-parameters or with a different architecture.

To run the validation script from the project root:
```sh
$ php validate.php
```

or

```sh
$ composer validate
```

### Wrap Up

- Natural Language Processing is the process of making sense of language using machine learning and other techniques
- One way to represent a document is by using a *bag-of-words* approach such as word counts or TF-IDF values
- Deep (Representation) Learning involves learning higher-order representations of the input data during training
- Neural Networks are a type of Deep Learning
- Neural Nets are composed of intermediate computational units called *hidden layers* that define the architecture of the network

## Original Dataset
See DATASET_README. For comments or questions regarding the dataset please contact [Andrew Maas](http://www.andrew-maas.net).

### References
[1] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).