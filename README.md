# Text Sentiment Analyzer using Rubix ML

This is a multi layer feed forward neural network for text sentiment classification (*positive* or *negative*) trained on 25,000 movie reviews from the [IMDB](https://www.imdb.com/) movie reviews website. The dataset also provides another 25,000 samples which we use to validate the model. This example project demonstrates text feature representation and deep learning using a type of neural network classifier called a Multi Layer Perceptron.

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
Our objective is to predict the sentiment (either *positive* or *negative*) of a blob of English text. We sometimes refer to this type of machine learning as Natural Language Processing (or *NLP* for short) because it involves making sense of language. The dataset provided to us contains 50,000 training and testing samples each consisting of a blob of English text describing a movie review. The samples have been pre labeled as being either positive or negative sentiment. By building a vocabulary of the most commonly used words, we can encode each blob of text as a fixed length vector of word counts so we can then feed the transformed samples into a neural network. Fortunately, the [Word Count Vectorizer](https://github.com/RubixML/RubixML#word-count-vectorizer) will take care of all of this. We then apply a [Delta TF-IDF](https://github.com/RubixML/RubixML#delta-tf-idf-transformer) (a supervised Term Frequency - Inverse Document Frequency weighting) transformer to weight each word by its importance to the outcome of its observed class label.

We will use a 5 layer neural network called a [Multi Layer Perceptron](https://github.com/RubixML/RubixML#multi-layer-perceptron) that is trained iteratively using mini batch Gradient Descent with Backpropagation. You can think of each hidden layer of the neural net as forming higher and higher order language concepts until finally it reaches the conclusion of either *positive* or *negative* sentiment. In addition, we'll cover non-linear [activation functions](https://github.com/RubixML/RubixML#activation-functions) as well as a regularization method called [Dropout](https://github.com/RubixML/RubixML#dropout).

### Training
Before we dive into the architecture of the network, let's first take care of loading the provided dataset into memory. The samples are given to us in single `.txt` files and organized into `pos` and `neg` folders. We'll use PHP's built in `glob()` function to loop through all the text files in each folder and add their contents to a samples array. We'll also add *positive* and *negative* labels as the training signal.

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

Next we define our estimator instance. Since our features are in raw text format, we'll need to convert them to continuous values in order for the neural net to understand and train effectively. We do so with a transformer Pipeline consisting of [HTML Stripper](https://github.com/RubixML/RubixML#html-stripper), [Text Normalizer](https://github.com/RubixML/RubixML#text-normalizer), [Word Count Vectorizer](https://github.com/RubixML/RubixML#word-count-vectorizer), and [TF-IDF Transformer](https://github.com/RubixML/RubixML#tf-idf-transformer). If you are unfamiliar with transformer pipelines see the [Credit Card Default](https://github.com/RubixML/Credit) tutorial using Logistic Regression and the Pipeline wrapper.

The Word Count Vectorizer is a common *bag of words* feature extractor that uses a fixed vocabulary and term counts to denote words that appear in a document. We elect to limit the vocabulary to 10,000 of the most frequent words that satisfy the criteria of appearing in at least 5 different documents. In this way, we limit the amount of *noise* words that enter the training set. Another common text feature representation are TF-IDF values which take the term counts from Word Count Vectorizer and weight them by their inverse document frequencies (IDFs) which can be interpreted as their *importance* within the corpus.

> **Note**: Word count and TF-IDF feature representations throw away all sentence structure. Despite that, they still work pretty well in practice.

The next thing we need to do is define the architecture of the network's hidden layers as the first hyper-parameter of the Multi Layer Perceptron base estimator. Each of the 5 layers consist of a [Dense](https://github.com/RubixML/RubixML#dense) layer of neurons and a non-linear [Activation](https://github.com/RubixML/RubixML#activation) layer with optional [Dropout](https://github.com/RubixML/RubixML#dropout) regularization. The first 2 layers use a [Leaky ReLU](https://github.com/RubixML/RubixML#leaky-relu) activation function while the last 2 use a parametric form of the Leaky ReLU called [PReLU](https://github.com/RubixML/RubixML#prelu) (for *Parametric* Rectified Linear Unit). The final layer is a [Multiclass](https://github.com/RubixML/RubixML#multiclass) output layer using a [Softmax](https://github.com/RubixML/RubixML#softmax) activation function which is the default output layer for the Multi Layer Perceptron classifier. We've found that the default architecture works pretty well for this problem but feel free to experiment and come up with your own architecture.

> **Note**: The depth of the network is determined as the number of weight layers which include the 4 Dense layers and the Multiclass output layer.

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
use Rubix\ML\NeuralNet\Initializers\He;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel(new Pipeline([
    new HTMLStripper(),
    new TextNormalizer(),
    new WordCountVectorizer(10000, 10),
    new TfIdfTransformer(),
], new MultiLayerPerceptron([
    new Dense(100),
    new Activation(new LeakyReLU(0.1)),
    new Dropout(0.2),
    new Dense(70),
    new Activation(new LeakyReLU(0.1)),
    new Dropout(0.2),
    new Dense(40),
    new PReLU(0.25),
    new Dense(10),
    new PReLU(0.25),
], 300, new Adam(0.0001), 1e-4),
    new Filesystem(MODEL_FILE)
);
```

Observe that there is a general pattern and order to the hidden layers of the network. [Dense](https://github.com/RubixML/RubixML#dense) layers linearly transform the input, then an [Activation](https://github.com/RubixML/RubixML#activation) layer applies a non-linear transformation, and the process repeats. Optionally we add a mild [Dropout](https://github.com/RubixML/RubixML#dropout) to the first two layers as a method to prevent overfitting. For the activations we are using two different types of Activation layers (one *parametric* and one *non-parametric*) with the [Leaky ReLU](https://github.com/RubixML/RubixML#leaky-relu) activation function. The last two hidden layers use a parametric form of the Leaky ReLU (called [PReLU](https://github.com/RubixML/RubixML#prelu)) activation function that learns the optimal amount of *leakage* to apply during training. Refer to the API Reference on [hidden layers](https://github.com/RubixML/RubixML#hidden-layers) for further reading.

The remaining hyper-parameters *batch size*, *optimizer* with *learning rate*, and *l2 regularization* amount will now be set. Batch size determines the number of training samples to run through the network at one time. The Gradient Descent optimizer determines the step size for each parameter in the network and most optimizers allow you to set a *learning rate* which controls the master step size. When setting the learning rate of an Optimizer, the important thing to note is that a learning rate that is too low will train slowly while a rate that is to high will prevent the network from learning at all. Lastly, we apply a small l2 penalty to the weights in the network such to help prevent them from overfitting the training data. For the full list of hyper-parameters, check out the [Multi Layer Perceptron](https://github.com/RubixML/RubixML#multi-layer-perceptron) API reference.

Lastly, we wrap the entire pipeline in a [Persistent Model](https://github.com/RubixML/RubixML#persistent-model) wrapper so we can save and load it later in a different process.

Now we're ready to train, but first let's set a logger instance so we can receive training updates through the console.

```php
use Rubix\ML\Other\Loggers\Screen;
use League\Csv\Writer;

$estimator->setLogger(new Screen('sentiment'));

$estimator->train($training);

$writer = Writer::createFromPath(PROGRESS_FILE, 'w+');
$writer->insertOne(['loss', 'score']);
$writer->insertAll(array_map(null, $estimator->steps(), $estimator->scores()));
```

Simply call `train()` with the training set we instantiated earlier and monitor the progress through the console. We also use the PHP League's [CSV Writer](https://csv.thephpleague.com/) to dump the loss and validation scores of each epoch during training to a CSV file for easy importing into a plotting application such as [Tableu](https://public.tableau.com/en-us/s/) or [Excel](https://products.office.com/en-us/excel).

> **Note**: Due to the high dimensionality of the TF-IDF input vectors and the width of the hidden layers this network has over 1 million individual weight parameters. Training may take a long time, however you will be able to monitor its progress using the Screen logger or any other PSR-3 compatible logger.

Finally, we prompt to save the model so we can use it later in our validation script.

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

On the map ...

### Cross Validation

On the map ...

### Wrap Up

On the map ...

## Original Dataset
See DATASET_README. For comments or questions regarding the dataset please contact [Andrew Maas](http://www.andrew-maas.net).

### References
[1] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).