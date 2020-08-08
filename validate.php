<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$samples = $labels = [];

foreach (['positive', 'negative'] as $label) {
    foreach (glob("test/$label/*.txt") as $file) {
        $samples[] = [file_get_contents($file)];
        $labels[] = $label;
    }
}

$dataset = Labeled::build($samples, $labels)->randomize()->take(10000);

$estimator = PersistentModel::load(new Filesystem('sentiment.model'));

$logger->info('Making predictions');

$predictions = $estimator->predict($dataset);

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $dataset->labels());

echo $results;

$results->toJSON()->write('report.json');

$logger->info('Report saved to report.json');