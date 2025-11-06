<?php
// Jalankan script Python
// $output = shell_exec("run_training.py 2>&1");
$output = shell_exec('python project-prediction-app-main/run_training.py');
// Tampilkan hasil di browser
echo "<pre>$output</pre>";
?>
