document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const chatLog = document.getElementById('chat-log');
    const questionInput = document.getElementById('question');
    const suggestionsContainer = document.getElementById('suggestions');

    function askQuestion(question) {
        if (!question) return;

        // Add user message to chat log
        addMessage(question, 'user');

        // Send question to server
        fetch('/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                'question': question
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log("Received data:", data); // Log the entire response

            // Add AI response to chat log
            if (data.answer) {
                addMessage(data.answer, 'ai');
            } else {
                addMessage("Sorry, I couldn't generate an answer.", 'ai');
            }

            // Update suggestions
            if (data.similar_questions) {
                updateSuggestions(data.similar_questions);
            }

            // Optionally, you can display the cluster information
            if (data.query_cluster !== undefined) {
                console.log("Assigned Cluster:", data.query_cluster);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage("An error occurred while processing your request.", 'ai');
        });

        // Clear input field
        questionInput.value = '';
    }

    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        askQuestion(questionInput.value);
    });

    function addMessage(text, sender) {
        console.log("Adding message:", text, sender);
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        messageDiv.innerHTML = `<p>${text}</p>`;
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight;
    }

    function updateSuggestions(newSuggestions) {
        suggestionsContainer.innerHTML = '';
        newSuggestions.forEach(suggestion => {
            const button = document.createElement('button');
            button.classList.add('suggestion-btn');
            button.textContent = suggestion;
            button.addEventListener('click', function() {
                askQuestion(this.textContent);
            });
            suggestionsContainer.appendChild(button);
        });
    }

    // Initial setup for suggestion buttons
    document.querySelectorAll('.suggestion-btn').forEach(button => {
        button.addEventListener('click', function() {
            askQuestion(this.textContent);
        });
    });
});