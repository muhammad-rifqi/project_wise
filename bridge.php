<?php
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST');
header("Access-Control-Allow-Headers: X-Requested-With");

// $output = shell_exec("run_training.py 2>&1");
$output = shell_exec('python project-prediction-app-main/run_training.py');
$input = file_put_contents('log.txt', $output);
if($input){
    $jsonData = file_get_contents('http://localhost:8080/project_wise/model/training_results.json');
    $encode = json_decode($jsonData,true);
    $logs = 'http://localhost:8080/project_wise/log.txt';
    $array = array(
        "success" => true,
        "data" => $encode,
        "log" => $logs
    );
    echo json_encode($array);
}else{
    echo json_encode(array("success" => "failed"));
}


?>
