<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\Transformers\HTMLStripper;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Transformers\TextNormalizer;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Classifiers\MultiLayerPerceptron;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use League\Csv\Writer;

const MODEL_FILE = 'sentiment.model';
const PROGRESS_FILE = 'progress.csv';

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ Text Sentiment Analyzer using Multi Layer Neural Network      ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

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
], 300, new Adam(0.0001), 1e-4)),
    new Filesystem(MODEL_FILE)
);

$estimator->setLogger(new Screen('sentiment'));

$estimator->train($training);

$writer = Writer::createFromPath(PROGRESS_FILE, 'w+');
$writer->insertOne(['loss', 'score']);
$writer->insertAll(array_map(null, $estimator->steps(), $estimator->scores()));

$estimator->prompt();
