<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\TextNormalizer;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Tokenizers\NGram;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Extractors\CSV;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$samples = $labels = [];

foreach (['positive', 'negative'] as $label) {
    foreach (glob("train/$label/*.txt") as $file) {
        $samples[] = [file_get_contents($file)];
        $labels[] = $label;
    }
}

$dataset = new Labeled($samples, $labels);

$estimator = new PersistentModel(
    new Pipeline([
        new TextNormalizer(),
        new WordCountVectorizer(10000, 2, 0.4, new NGram(1, 2)),
        new TfIdfTransformer(),
        new ZScaleStandardizer(),
    ], new MultilayerPerceptron([
        new Dense(100),
        new Activation(new LeakyReLU()),
        new Dense(100),
        new Activation(new LeakyReLU()),
        new Dense(100, 0.0, false),
        new BatchNorm(),
        new Activation(new LeakyReLU()),
        new Dense(50),
        new PReLU(),
        new Dense(50),
        new PReLU(),
    ], 256, new AdaMax(0.0001))),
    new Filesystem('sentiment.rbx', true)
);

$estimator->setLogger($logger);

$estimator->train($dataset);

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->steps());

$logger->info('Progress saved to progress.csv');

if (strtolower(trim(readline('Save this model? (y|[n]): '))) === 'y') {
    $estimator->save();
}
