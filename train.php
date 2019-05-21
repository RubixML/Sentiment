<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Other\Tokenizers\Word;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\Other\Tokenizers\NGram;
use Rubix\ML\Transformers\HTMLStripper;
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\Transformers\TextNormalizer;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use Rubix\ML\Transformers\TfIdfTransformer;
use Rubix\ML\Transformers\ZScaleStandardizer;
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

echo 'Loading data into memory ...' . PHP_EOL;

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
        new Dense(30),
        new PReLU(),
    ], 200, new AdaMax(0.00005))),
    new Filesystem(MODEL_FILE, true)
);

$estimator->setLogger(new Screen('sentiment'));

$estimator->train($training);

$writer = Writer::createFromPath(PROGRESS_FILE, 'w+');
$writer->insertOne(['loss', 'score']);
$writer->insertAll(array_map(null, $estimator->steps(), $estimator->scores()));

echo 'Progress saved to ' . PROGRESS_FILE . PHP_EOL;

if (strtolower(trim(readline('Save this model? (y|[n]): '))) === 'y') {
    $estimator->save();
}
