<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Datasets\Unlabeled;

ini_set('memory_limit', '-1');

$transformer = PersistentTransformer::load(new Filesystem('transformer.rbx'));

$estimator = PersistentModel::load(new Filesystem('model.rbx'));

while (empty($text)) $text = readline("Enter some text to analyze:\n");

$dataset = new Unlabeled([
    [$text],
]);

$dataset->apply($transformer);

$predictions = $estimator->predict($dataset);

$prediction = current($predictions);

echo "The sentiment is: $prediction" . PHP_EOL;
