<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

ini_set('memory_limit', '-1');

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ Text Sentiment Analyzer using Multi Layer Perceptron          ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

$estimator = PersistentModel::load(new Filesystem('sentiment.model'));

while (empty($text)) $text = readline("Enter some text to analyze:\n");

$prediction = $estimator->predictSample([$text]);

echo "The sentiment is: $prediction" . PHP_EOL;
