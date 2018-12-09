<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Other\Functions\Argmax;

const MODEL_FILE = 'sentiment.model';

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ Text Sentiment Analyzer using Multi Layer Neural Network      ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

$text = readline('Enter text to analyze: ');

$dataset = Unlabeled::build([$text]);

$estimator = PersistentModel::load(new Filesystem(MODEL_FILE));

$probabilities = $estimator->proba($dataset)[0];

$prediction = Argmax::compute($probabilities);

echo PHP_EOL; 

echo 'Prediction :' . $prediction . PHP_EOL;

var_dump($probabilities);
