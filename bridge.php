<?php
// $output = shell_exec("run_training.py 2>&1");
$output = shell_exec('python project-prediction-app-main/run_training.py');
$input = file_put_contents('log.txt', $output);
if($input){
    $jsonData = file_get_contents('http://localhost/project-prediction-app-main/model/training_results.json');
    $encode = json_decode($jsonData,true);
    $array = array(
        "success" => true,
    );
}


?>
