<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\TextNormalizer;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Tokenizers\NGram;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\Swish;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU;
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
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
    base: new Pipeline([
        new TextNormalizer(),
        new WordCountVectorizer(10240, 2, 0.4, new NGram(1, 2)),
        new FloatTypeConverter(),
        new TfIdfTransformer(sublinear: true),
        new ZScaleStandardizer(),
    ], new MultilayerPerceptron(
        hiddenLayers: [
            new Dense(128),
            new Activation(new SiLU()),
            new Dense(128),
            new Activation(new SiLU()),
            new Dense(128, 0.0, false),
            new BatchNorm(),
            new Activation(new SiLU()),
            new Dense(64),
            new Swish(),
            new Dense(64),
            new Swish(),
            new Dense(2),
        ], 
        batchSize: 32,
        gradientAccumulationSteps: 4,
        optimizer: new AdaMax(new Constant(0.0001)),
        maxGradientNorm: 1.0,
        epochs: 100,
        minChange: 1e-5,
        evalInterval: 1,
        window: 10,
        holdOut: 0.1
    )),
    persister: new Filesystem('sentiment.rbx', true)
);

$estimator->setLogger($logger);

$estimator->train($dataset);

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->steps());

$logger->info('Progress saved to progress.csv');

if (strtolower(trim(readline('Save this model? (y|[n]): '))) === 'y') {
    $estimator->save();
}
