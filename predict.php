<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Datasets\Unlabeled;

ini_set('memory_limit', '-1');

$estimator = PersistentModel::load(new Filesystem('sentiment.rbx'));

while (empty($text)) $text = readline("Enter some text to analyze:\n");

$dataset = new Unlabeled([
    [$text],
]);

$prediction = current($estimator->predict($dataset));

echo "The sentiment is: $prediction" . PHP_EOL;
