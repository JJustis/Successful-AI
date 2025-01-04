<?php
$data = file_get_contents('php://input');
$result = file_put_contents('training.json', $data);

if ($result === false) {
    http_response_code(500);
    echo json_encode(['error' => 'Failed to save data']);
} else {
    http_response_code(200);
    echo json_encode(['success' => true]);
}
?>