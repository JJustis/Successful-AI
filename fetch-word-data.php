<?php
header('Content-Type: application/json');

$host = 'localhost';
$dbname = 'reservesphp';
$username = 'root';
$password = '';

try {
    $pdo = new PDO("mysql:host=$host;dbname=$dbname", $username, $password);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);

    $word = $_GET['word'] ?? '';
    if (empty($word)) {
        throw new Exception('No word provided');
    }

    $word = rtrim(strtolower($word), '.!?,:;');

    $stmt = $pdo->prepare("SELECT definition, wiki, type FROM word WHERE LOWER(word) = :word");
    $stmt->execute(['word' => $word]);
    $result = $stmt->fetch(PDO::FETCH_ASSOC);

    if ($result) {
        echo json_encode($result);
    } else {
        echo json_encode(['error' => 'Word not found', 'word' => $word]);
    }
} catch (Exception $e) {
    echo json_encode(['error' => $e->getMessage(), 'word' => $word ?? 'unknown']);
}
?>