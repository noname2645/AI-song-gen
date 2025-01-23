document.getElementById('generate-btn').addEventListener('click',()=>{
    const prompt = document.getElementById('prompt').value;
    fetch('http://localhost:3000/generate',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({prompt})
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('output').innerText = data.song || "Error generating song.";
    })
    .catch(err => {
        document.getElementById('output').innerText = "Something went wrong!";
    });
});