<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\PersistentModel;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

const MODEL_FILE = 'sentiment.model';
const REPORT_FILE = 'report.json';

echo '╔═══════════════════════════════════════════════════════════════╗' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '║ Cross Validation Report for Text Sentiment Analyzer           ║' . PHP_EOL;
echo '║                                                               ║' . PHP_EOL;
echo '╚═══════════════════════════════════════════════════════════════╝' . PHP_EOL;
echo PHP_EOL;

echo 'Loading data into memory ...' . PHP_EOL;

$samples = $labels = [];

foreach (glob(__DIR__ . '/test/pos/*.txt') as $file) {
        $samples[] = [file_get_contents($file)];
        $labels[] = 'positive';
}

foreach (glob(__DIR__ . '/test/neg/*.txt') as $file) {
        $samples[] = [file_get_contents($file)];
        $labels[] = 'negative';
}

$testing = Labeled::build($samples, $labels)->randomize()->take(15000);

$estimator = PersistentModel::load(new Filesystem(MODEL_FILE));

$predictions = $estimator->predict($testing);

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $testing->labels());

file_put_contents(REPORT_FILE, json_encode($results, JSON_PRETTY_PRINT));

echo 'Report saved to ' . REPORT_FILE . PHP_EOL;
