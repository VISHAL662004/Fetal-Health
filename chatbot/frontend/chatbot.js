async function sendMessage() {
    const inputBox = document.getElementById("user-input");
    const chatWindow = document.getElementById("chat-window");
    const userMessage = inputBox.value;
    
    chatWindow.innerHTML += `<p><b>You:</b> ${userMessage}</p>`;
    inputBox.value = "";

    const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({"message": userMessage})
    });

    const data = await response.json();
    chatWindow.innerHTML += `<p><b>Bot:</b> ${data.response}</p>`;
    chatWindow.scrollTop = chatWindow.scrollHeight;
}
