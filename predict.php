<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Persisters\Filesystem;

const MODEL_FILE = 'sentiment.model';

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ Text Sentiment Analyzer using Multi Layer Neural Network      ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

$estimator = PersistentModel::load(new Filesystem(MODEL_FILE));

$text = readline('Enter text to analyze: ');

$dataset = Unlabeled::build([$text]);

$probabilities = $estimator->proba($dataset);

echo PHP_EOL . 'Probabilities: ' . PHP_EOL;

var_dump($probabilities[0]);
