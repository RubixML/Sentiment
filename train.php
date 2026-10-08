<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Transformers\Pipeline;
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

$datasets = [];

foreach (['train', 'test'] as $split) {
    $samples = $labels = [];

    foreach (['positive', 'negative'] as $label) {
        foreach (glob("$split/$label/*.txt") as $file) {
            $samples[] = [file_get_contents($file)];
            $labels[] = $label;
        }
    }

    $datasets[] = new Labeled($samples, $labels);
}

[$training, $testing] = $datasets;

$transformer = new PersistentTransformer(
    base: new Pipeline([
        new TextNormalizer(),
        new WordCountVectorizer(10240, 2, 0.4, new NGram(1, 2)),
        new FloatTypeConverter(),
        new TfIdfTransformer(sublinear: true),
        new ZScaleStandardizer(),
    ]),
    persister: new Filesystem('transformer.rbx', true)
);

$estimator = new PersistentModel(
    base: new MultilayerPerceptron(
        hiddenLayers: [
            new Dense(256),
            new Activation(new SiLU()),
            new Dense(256, bias: false),
            new BatchNorm(),
            new Activation(new SiLU()),
            new Dense(256),
            new Activation(new SiLU()),
            new Dense(128, bias: false),
            new BatchNorm(),
            new Swish(),
            new Dense(128),
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
        window: 10
    ),
    persister: new Filesystem('model.rbx', true)
);

$estimator->setLogger($logger);

$logger->info('Preprocessing dataset');

$transformer->fit($training);

$transformer->save();

$training->apply($transformer);
$testing->apply($transformer);

$estimator->setValidationDataset($testing);

$estimator->train($training);

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->progress());

$logger->info('Progress saved to progress.csv');

if (strtolower(trim(readline('Save this model? (y|[n]): '))) === 'y') {
    $estimator->save();
}
